"""The orchestrator.

It owns the order of operations, the human confirmation gate, task tracking and
the conversation. It does not compute anything: every identifier, number and
status it reports came back from one of the two MCP servers through a named
client method.

A request is handled like this:

1. let the model choose question or task, then validate task inputs locally;
2. check the workflow gate for the operation that intent implies — this is what
   refuses an unconfirmed watershed, an unvalidated input or an unplanned run;
3. call the MCP tool, logging the call and its outcome;
4. record the facts the server returned, advance the state on the transition
   table, and save the session;
5. describe the outcome.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from . import DISCLAIMER_ZH, domain_knowledge
from .artifacts import ArtifactDownload, ArtifactService, DownloadError
from .clarification import Clarification, ask as ask_one_question, blocked as blocked_question, describe_recorded
from .config import AgentSettings, load_settings
from .conversation_responder import ConversationResponder
from .dialogue_router import ENTITY_FIELDS, REQUIRED_ENTITIES, DialogueDecision, DialogueRouter, RouteKind, validated_payload
from .engineering_knowledge import asks_current_result, asks_general_concept, is_system_question
from .evaluation import TOPICS as TERM_TOPICS
from .llm import build_llm_client
from .intent import Intent, IntentParser, ParsedIntent, awaits_panel_approval
from .mcp_clients import (
    AgentMCPError,
    HydroMCPClient,
    SpatialMCPClient,
)
from .models import (
    AgentSession,
    ArtifactDescriptor,
    HydroRunRequestPayload,
    RainfallRequestPayload,
    VisualizationResult,
    minimise_plan,
)
from .response_writer import (
    describe_artifact_choices,
    describe_artifact_ready,
    describe_cancellation,
    describe_capabilities,
    describe_catalog,
    describe_confirmation,
    describe_gate_error,
    describe_import,
    describe_input_validation,
    describe_mcp_error,
    describe_model_inputs,
    describe_missing_fields,
    describe_no_figure_artifacts,
    describe_plan,
    describe_rainfall_package,
    describe_review,
    describe_run_started,
    describe_run_status,
    describe_scope,
    describe_swmm_bundle_ready,
    describe_swmm_model,
    describe_unknown,
    describe_visualization_ready,
    sanitize,
)
from .response_rewriter import CanonicalReply, Fact, ResponseRewriter
from .scientific_policy import EvidenceLedger, audit_reply
from .session_store import SessionStore, sanitize_user_text
from .state_machine import (
    IllegalTransition,
    Operation,
    WorkflowError,
    WorkflowState,
    advance,
    require_operation,
)


#: How a finished hydro run maps onto a session state.
RUN_STATUS_TO_STATE = {
    "succeeded": WorkflowState.HYDRO_SUCCEEDED,
    "qa_failed": WorkflowState.HYDRO_QA_FAILED,
    "failed": WorkflowState.HYDRO_FAILED,
    "cancelled": WorkflowState.HYDRO_CANCELLED,
}

ACTIVE_RUN_STATUSES = frozenset({"queued", "running"})


@dataclass
class AgentReply:
    """What the CLI renders for one user turn."""

    text: str
    session: AgentSession
    #: Image payloads to display as-is; never converted to JSON by the agent.
    images: list[dict[str, Any]] = field(default_factory=list)
    #: Figures this turn produced, with the public descriptors of the real files
    #: behind them. Kept alongside ``images`` rather than replacing it: the CLI,
    #: the existing tests and older clients read ``images``.
    visualizations: list[Any] = field(default_factory=list)
    #: Structured facts behind the reply, for the event log and for tests.
    facts: dict[str, Any] = field(default_factory=dict)
    kind: str = "message"
    ok: bool = True
    next_step: str = ""
    #: Problems the audit found in the text the model produced, if any. Empty
    #: means the reply the user sees is the audited one.
    audit: list[str] = field(default_factory=list)
    #: True when the language model's phrasing was kept.
    rewritten: bool = False
    #: Transient map metadata for the interactive Web picker; never persisted.
    outlet_map: dict[str, Any] | None = None
    answer_source: str = "mcp_evidence"
    audit_scope: str = "computation"

    def with_session(self, session: AgentSession) -> "AgentReply":
        self.session = session
        return self


from .task_workspace import TaskWorkspaceMixin


from .product_cleanup import ProductCleanupMixin, record_created_product


class SlopeEvaluationAgent(TaskWorkspaceMixin, ProductCleanupMixin):
    """Orchestrates the two MCP servers for one conversation."""

    def __init__(
        self,
        settings: AgentSettings | None = None,
        *,
        spatial: SpatialMCPClient | None = None,
        hydro: HydroMCPClient | None = None,
        store: SessionStore | None = None,
        parser: IntentParser | None = None,
        rewriter: ResponseRewriter | None = None,
        router: DialogueRouter | None = None,
        responder: ConversationResponder | None = None,
        locator=None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.logger = logger or logging.getLogger("slope_agent")
        self.store = store or SessionStore(self.settings.sessions)
        self.spatial = spatial or SpatialMCPClient(self.settings.mcp.spatial, self.logger)
        self.hydro = hydro or HydroMCPClient(self.settings.mcp.hydro, self.logger)
        # One client serves routing, grounded rephrasing and general knowledge.
        # Without a configuration every caller falls back to built-in behaviour.
        self.llm = build_llm_client(self.settings.llm, self.logger)
        self.parser = parser or IntentParser(llm=self.llm, logger=self.logger)
        self.rewriter = rewriter or ResponseRewriter(self.llm, self.logger)
        # Share deterministic entity checks and explicit human decisions;
        # the model chooses the route for every ordinary message.
        self.router = router or DialogueRouter(llm=self.llm,
                                               settings=self.settings.dialogue,
                                               rules=self.parser.rules,
                                               logger=self.logger)
        self.responder = responder or ConversationResponder(
            llm=self.llm if self.settings.dialogue.allow_general_llm else None,
            recent_turns=self.settings.dialogue.recent_turns, logger=self.logger)
        from .geolocation import GeographicLocator
        self.locator = locator or GeographicLocator(self.llm, self.logger)
        # What this conversation may download. It holds no state of its own: the
        # descriptors live in the session, so a reopened conversation offers
        # exactly the downloads it was given.
        self.artifacts = ArtifactService(self.settings, self.hydro, self.logger)

    # ------------------------------------------------------------- lifecycle
    async def open(self) -> None:
        """Connect both servers. Either failure leaves the agent usable."""
        for client in (self.spatial, self.hydro):
            try:
                await client.connect()
            except AgentMCPError as exc:
                self.logger.warning("%s", exc.user_message())

    async def close(self) -> None:
        for client in (self.spatial, self.hydro):
            await client.close()

    async def __aenter__(self) -> "SlopeEvaluationAgent":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def new_session(self, *, language: str | None = None) -> AgentSession:
        return self.store.create(language=language or self.settings.language)

    # ---------------------------------------------------------------- events
    def _events(self, session: AgentSession):
        return self.store.events(session.session_id)

    async def _call(self, session: AgentSession, client, method: str, **arguments) -> dict:
        """Call one named client method, logging before and after."""
        events = self._events(session)
        events.append("tool_call", tool=method, arguments=_log_safe(arguments))
        try:
            payload = await getattr(client, method)(**arguments)
        except Exception as exc:  # noqa: BLE001 - every failure is mapped below
            events.append("tool_error", tool=method, error=type(exc).__name__,
                          message=str(exc)[:300])
            raise
        events.append("tool_result", tool=method,
                      ok=payload.get("ok"), status=payload.get("status"))
        record_created_product(self.store, session.session_id, method, payload)
        return payload

    def _advance(self, session: AgentSession, target: WorkflowState, reason: str) -> None:
        if session.state != target.value:
            advance(session, target, reason=reason, events=self._events(session))

    async def _finish(self, session: AgentSession, text: str, *,
                      facts: dict[str, Any] | None = None,
                      ledger: EvidenceLedger | None = None,
                      rewrite_response: bool = True,
                      clear_error: bool = True,
                      **kwargs) -> AgentReply:
        """Return the audited reply for this turn.

        The deterministic text is the canonical one. The model may rephrase it,
        and whatever comes back is audited against the ledger: a rewrite that
        invents a number, changes a unit or asserts a conclusion is discarded and
        the canonical text is what the user sees.
        """
        if clear_error:
            session.clear_error()
        self.store.save(session)
        canonical_text = sanitize(text)
        evidence = ledger or self._ledger_for(session, facts or {})
        canonical = CanonicalReply(
            intent=session.intent or "",
            status=kwargs.get("kind", "message"),
            canonical_text=canonical_text,
            facts=_canonical_facts(evidence),
        )
        canonical_problems = audit_reply(canonical_text, evidence)
        if canonical_problems:
            # A deterministic template is not exempt from the scientific gate.
            # Returning it merely because a model rewrite failed would make the
            # audit fail-open exactly when the evidence mapping is broken.
            self.logger.error("确定性回复未通过审计：%s", canonical_problems[:3])
            self._events(session).append(
                "reply_audit", problems=canonical_problems[:5], rewritten=False,
                canonical_failed=True)
            safe_text = ("The reply failed its consistency check and numerical results were withheld. "
                         "Keep the session and verify the evidence before retrying.")
            kwargs["kind"] = "error"
            kwargs["ok"] = False
            return AgentReply(text=safe_text, session=session, facts=facts or {},
                              audit=canonical_problems, rewritten=False, **kwargs)
        try:
            candidate = (await self._rewrite_reply(canonical=canonical, evidence=evidence)
                         if rewrite_response else canonical_text)
        except Exception as exc:  # noqa: BLE001 - a rewrite failure is never fatal
            self.logger.info("回复改写失败，使用确定性文本：%s", exc)
            candidate = canonical_text
        if not isinstance(candidate, str) or not candidate.strip():
            candidate = canonical_text
        problems = audit_reply(candidate, evidence)
        if problems:
            self.logger.info("回复未通过审计，回退确定性文本：%s", problems[:3])
            candidate = canonical_text
        self._events(session).append(
            "reply_audit", problems=problems[:5],
            rewritten=bool(candidate != canonical_text))
        return AgentReply(text=candidate, session=session, facts=facts or {},
                          audit=problems, rewritten=candidate != canonical_text, **kwargs)

    async def _rewrite_reply(self, *, canonical: CanonicalReply,
                             evidence: EvidenceLedger) -> str:
        """Rephrase the canonical reply. Overridden in tests to inject a liar."""
        outcome = await self.rewriter.rewrite(canonical, evidence)
        if outcome.fallback_reason:
            self._pending_rewrite_note = outcome.fallback_reason
        return outcome.text

    async def _finish_knowledge(self, session: AgentSession, answer, parsed) -> AgentReply:
        """Knowledge is labelled as such and never certified as MCP evidence."""
        self._events(session).append("knowledge_reply", **answer.log_safe())
        self.store.save(session)
        return AgentReply(text=answer.text, session=session, kind="explanation",
                          answer_source=answer.answer_source,
                          audit_scope=('task_context' if answer.answer_source == 'saved_task_context' else 'general_knowledge'),
                          facts={"chat_topic": answer.topic, "turn_intent": parsed.intent.value,
                                 "knowledge_fallback": answer.fallback_reason})

    def _ledger_for(self, session: AgentSession,
                    facts: dict[str, Any]) -> EvidenceLedger:
        """Everything this reply is allowed to state: its own facts plus the run's."""
        ledger = EvidenceLedger.from_evidence(facts)
        assessment = self.store.load_assessment(session.session_id)
        if assessment:
            ledger.extend(EvidenceLedger.from_assessment(assessment))
        return ledger

    def _fail(self, session: AgentSession, error: Exception, *,
              code: str | None = None) -> AgentReply:
        if isinstance(error, WorkflowError):
            message = describe_gate_error(error, session)
        else:
            message = describe_mcp_error(error)
        session.note_error(
            code=code or type(error).__name__,
            message=message.splitlines()[0] if message else type(error).__name__,
            next_step=getattr(error, "next_step", ""),
        )
        self.store.save(session)
        return AgentReply(text=message, session=session, kind="error", ok=False,
                          next_step=getattr(error, "next_step", ""))

    # ------------------------------------------------------------------ turn
    async def handle(self, session: AgentSession, text: str) -> AgentReply:
        """Handle one user turn: route, validate, dispatch, audit, persist.

        Routing decides *what kind of turn* this is; it never executes anything.
        A workflow action goes to the same handlers and the same gates as
        before — the router proposes, and the state machine still disposes.
        """
        # A pasted machine path or credential is neither a scientific input nor
        # something the optional language model should ever see.
        text = sanitize_user_text(text)
        # Restored sessions from earlier releases incorrectly waited for a
        # human confirmation even after a failed spatial QA verdict.
        if (session.state == WorkflowState.AWAITING_WATERSHED_CONFIRMATION.value
                and session.watershed_qa_passed is False):
            self._advance(session, WorkflowState.WATERSHED_QA_FAILED, "恢复质量检查失败的划分")
        session.record_turn(
            "user", text,
            maximum_turns=self.settings.sessions.max_conversation_turns,
            maximum_chars=self.settings.sessions.max_text_chars_per_turn,
        )
        decision = await self.router.decide(text, session)
        parsed, problems = self.router.to_parsed_intent(decision, text, session)
        self._record_routing(session, decision, parsed, problems)
        # What the session held before this turn, so a turn that would replace
        # an input the artefacts were built from can be taken back cleanly.
        before = session.model_dump(mode="json")
        recorded: dict[str, Any] = {}
        if decision.route in {RouteKind.WORKFLOW_ACTION, RouteKind.CLARIFICATION}:
            if decision.intent is Intent.UNKNOWN:
                recorded, _ = validated_payload(decision.grounded_candidates(text))
                for name in DialogueDecision.conflicts(decision.grounded_candidates(text)):
                    recorded.pop(name, None)
            else:
                recordable = parsed
                if (decision.route is RouteKind.CLARIFICATION
                        and decision.intent is Intent.SET_RAINFALL_PERIOD
                        and getattr(self.router,"last_source",None)=="model"):
                    # Unresolved event candidates belong to the pending task,
                    # not to active inputs already associated with results.
                    recordable = parsed.model_copy(update={key:None for key in
                        ("date","start_datetime","end_datetime","rainfall_mode","lookback_years","end_year")})
                recorded = self._remember_entities(session, recordable)
            self._remember_pending_workflow(session, decision, parsed, text)
        reply = await self._dispatch_decision(session, decision, parsed, recorded, before, text)
        if (reply.ok and not reply.facts.get("pending_user_control")
                and session.last_route == RouteKind.WORKFLOW_ACTION.value
                and ((parsed.intent is Intent.SET_RAINFALL_PERIOD and session.rainfall_package_id)
                     or parsed.intent is Intent.CANCEL_RUN)):
            session.pending_workflow = {}
            session.pending_question = None
        session.record_turn(
            "agent", reply.text,
            tools=[],
            answer_source=reply.answer_source,
            maximum_turns=self.settings.sessions.max_conversation_turns,
            maximum_chars=self.settings.sessions.max_text_chars_per_turn,
        )
        session.refresh_summary()
        self.store.save(session)
        return reply

    def _record_routing(self, session: AgentSession, decision: DialogueDecision,
                        parsed, problems: list[str]) -> None:
        """Log what the router decided, without logging what it was shown.

        Field names, counts and reasons only: the event log is for auditing the
        routing, and a value the user typed belongs in the session, not twice in
        the log. No prompt or credential is ever recorded here.
        """
        events = self._events(session)
        events.append("dialogue_route", **decision.log_safe(), source=getattr(self.router, "last_source", "rules"))
        if getattr(self.router, "last_diagnostic", None):
            events.append("dialogue_route_diagnostic", **self.router.last_diagnostic)
        for field in sorted({item.field for item in decision.candidates}):
            events.append("entity_candidate", field=field)
        for reason in problems[:5]:
            events.append("entity_rejected", reason=reason[:200])
        # The chat routes must not overwrite the last workflow intent: the
        # difference between "what we were doing" and "what was said last"
        # matters when the user returns to the task.
        session.last_route = decision.route.value
        if decision.route is RouteKind.WORKFLOW_ACTION:
            session.intent = parsed.intent.value
            events.append("intent", intent=parsed.intent.value,
                          confidence=parsed.confidence, source=parsed.source)

    @staticmethod
    def _remember_pending_workflow(session, decision, parsed, text):
        if decision.intent is Intent.UNKNOWN:
            supplied, _ = validated_payload(decision.grounded_candidates(text))
            if not supplied:
                return
        elif decision.intent in REQUIRED_ENTITIES:
            supplied = {name: getattr(parsed, name) for name in ENTITY_FIELDS
                        if getattr(parsed, name, None) is not None}
        else:
            return
        intent = decision.intent.value
        old = session.pending_workflow
        compatible = (old.get("intent") in {intent, "unknown"} or intent == "unknown" or
                      {old.get("intent"), intent} <= {"start_assessment", "select_outlet"})
        if intent == "unknown":
            intent = old.get("intent", intent)
        entities = dict(old.get("entities", {})) if compatible else {}
        conflicts = DialogueDecision.conflicts(decision.grounded_candidates(text))
        unresolved = (set(old.get("unresolved_fields", [])) - supplied.keys()) if compatible else set()
        for name in conflicts:
            supplied.pop(name, None)
            entities.pop(name, None)
        location_fields = {"administrative_area", "river_name", "longitude", "latitude"}
        if location_fields & supplied.keys():
            for name in location_fields:
                entities.pop(name, None)
            unresolved -= location_fields
        unresolved.update(conflicts)
        entities.update(supplied)
        session.pending_workflow = {"intent": intent, "entities": entities,
                                    "user_request": old.get("user_request", text[:1000]) if compatible else text[:1000],
                                    "unresolved_fields": sorted(unresolved),
                                    "rainfall_needs_clarification": bool(
                                        decision.route is RouteKind.CLARIFICATION and
                                        decision.intent is Intent.SET_RAINFALL_PERIOD)}

    @staticmethod
    def _remember_entities(session: AgentSession, parsed) -> dict[str, Any]:
        """Carry entities forward across turns.

        What a user says early ("分析大宁河 2024-07-30 全天") has to survive
        until the step that needs it, which may be several turns later.

        Returns the values this turn established that the session did not
        already hold, so a clarification can say them back without claiming
        credit for what the user said three turns ago.
        """
        fresh: dict[str, Any] = {}
        if parsed.intent not in {Intent.START_ASSESSMENT,Intent.SELECT_OUTLET,Intent.SET_RAINFALL_PERIOD,Intent.UNKNOWN}:
            return fresh
        location_input = parsed.intent in {Intent.START_ASSESSMENT,Intent.SELECT_OUTLET,Intent.UNKNOWN}
        if location_input and parsed.administrative_area and parsed.administrative_area != session.administrative_area:
            fresh["administrative_area"] = parsed.administrative_area
        if location_input and parsed.river_name and parsed.river_name != session.river_name:
            session.river_name = parsed.river_name
            session.location_hint = None
            fresh["river_name"] = parsed.river_name
        if location_input and parsed.longitude is not None and parsed.latitude is not None:
            outlet = session.requested_outlet or session.selection_context.get("map_center") or {}
            if (outlet.get("longitude"), outlet.get("latitude")) != (
                    parsed.longitude, parsed.latitude):
                fresh["longitude"] = parsed.longitude
                fresh["latitude"] = parsed.latitude
        rainfall: dict[str, Any] = {}
        if parsed.rainfall_mode:
            rainfall["mode"] = parsed.rainfall_mode
        if parsed.date:
            rainfall["date"] = parsed.date
        if parsed.start_datetime:
            rainfall["start_datetime"] = parsed.start_datetime
        if parsed.end_datetime:
            rainfall["end_datetime"] = parsed.end_datetime
        if parsed.lookback_years:
            rainfall["lookback_years"] = parsed.lookback_years
        if parsed.end_year:
            rainfall["end_year"] = parsed.end_year
        if parsed.station_count_mode == "auto":
            rainfall["station_count"] = None
        elif parsed.station_count is not None:
            rainfall["station_count"] = parsed.station_count
        if rainfall:
            stored = session.rainfall_request or {}
            for key, value in rainfall.items():
                if stored.get(key) != value:
                    fresh["rainfall_mode" if key == "mode" else key] = value
            stored = dict(stored)
            if rainfall.get("mode") == "historical_max":
                for key in ("date","start_datetime","end_datetime"):
                    stored.pop(key,None)
            elif rainfall.get("date"):
                if not (rainfall.get("start_datetime") and rainfall.get("end_datetime")):
                    stored.pop("start_datetime",None);stored.pop("end_datetime",None)
            if ((rainfall.get("start_datetime") or rainfall.get("end_datetime"))
                    and not (rainfall.get("date") and rainfall.get("start_datetime") and rainfall.get("end_datetime")
                             and _is_whole_day(rainfall["start_datetime"], rainfall["end_datetime"], rainfall["date"]))):
                stored.pop("date",None)
                rainfall.pop("date",None)
            session.rainfall_request = {**stored, **rainfall}
        return fresh

    @staticmethod
    def _rainfall_is_specified(session: AgentSession) -> bool:
        if session.pending_workflow.get("rainfall_needs_clarification"):
            return False
        request = session.rainfall_request or {}
        if request.get("mode") == "historical_max":
            return True
        if request.get("date"):
            return True
        return bool(request.get("start_datetime") and request.get("end_datetime"))

    async def _dispatch(self, session: AgentSession, parsed, text: str) -> AgentReply:
        handlers = {
            Intent.START_ASSESSMENT: self._handle_start_assessment,
            Intent.RESTORE_TASK: self._handle_restore_task,
            Intent.SELECT_OUTLET: self._handle_select_outlet,
            Intent.CONFIRM_WATERSHED: self._handle_confirm,
            Intent.REJECT_WATERSHED: self._handle_reject,
            Intent.SET_RAINFALL_PERIOD: self._handle_set_rainfall,
            Intent.START_HYDRO_RUN: self._handle_start_run,
            Intent.CHECK_STATUS: self._handle_check_status,
            Intent.CANCEL_RUN: self._handle_cancel,
            Intent.SHOW_RESULT: self._handle_show_result,
            Intent.SHOW_VISUALIZATION: self._handle_show_visualization,
            Intent.DOWNLOAD_ARTIFACT: self._handle_download_artifact,
            Intent.ASK_EXPLANATION: self._handle_explain,
        }
        handler = handlers.get(parsed.intent)
        if handler is None:
            # Must be awaited: returning the coroutine would hand an un-awaited
            # object back to the caller instead of a reply.
            return await self._handle_unknown(session, parsed, text)
        try:
            return await handler(session, parsed, text)
        except WorkflowError as exc:
            # A gate refusal means something else has to happen first. Saying
            # which thing is a question, not an error, so it is asked naturally
            # instead of being reported as a failure of the user's request.
            return await self._handle_blocked(session, exc, parsed)
        except AgentMCPError as exc:
            return self._fail(session, exc)
        except IllegalTransition as exc:
            return self._fail(session, exc)

    async def _dispatch_decision(self, session: AgentSession, decision: DialogueDecision,
                                 parsed, recorded: dict[str, Any],
                                 before: dict[str, Any], text: str) -> AgentReply:
        """Send the turn down the one channel the router chose."""
        try:
            from .historical_tasks import is_history_question, history_answer
            historical_read = (decision.route is RouteKind.KNOWLEDGE_CHAT or parsed.intent in {
                Intent.SHOW_RESULT, Intent.CHECK_STATUS, Intent.ASK_EXPLANATION} or
                (decision.route is RouteKind.CLARIFICATION and re.search(r"查询|查看|看看|区分|对比|比较|报告|列出|告诉|最小|参数", text)))
            historical_run = any(run != session.hydro_run_id for run in re.findall(
                r"(?<![A-Za-z0-9_-])(?:hydro-)?run-[A-Za-z0-9_-]+", text))
            if historical_read and (is_history_question(text) or historical_run):
                # Unknown foreign runs remain refused; an owned saved run may
                # be read without first restoring its inputs into the session.
                from .historical_tasks import select_revisions
                if historical_run and not is_history_question(text) and not select_revisions(self.store, session.session_id, text):
                    return await self._finish(session, '你指定的运行未与当前会话关联，不能读取其结果。',
                                              kind='question', ok=False, rewrite_response=False)
                answer = history_answer(self.store, session, text)
                return await self._finish(session, **answer,
                                          ledger=EvidenceLedger.from_evidence(answer['facts']),
                                          rewrite_response=False, clear_error=False,
                                          answer_source="saved_task_evidence")
            # A requested opaque run id cannot silently mean the current run.
            # This is a reference check within the existing routing contract,
            # not permission to switch sessions or query another user's run.
            cited_runs = set(re.findall(r'(?<![A-Za-z0-9_-])(?:hydro-)?run-[A-Za-z0-9_-]+', text))
            reading = (decision.route is RouteKind.KNOWLEDGE_CHAT or parsed.intent in {
                Intent.SHOW_RESULT, Intent.SHOW_VISUALIZATION, Intent.DOWNLOAD_ARTIFACT,
                Intent.CHECK_STATUS})
            if reading and len(cited_runs) == 1 and session.hydro_run_id not in cited_runs:
                return await self._finish(session,
                    '你指定的运行未与当前会话关联，当前不能读取或确认其结果。请确认任务编号或切换到对应任务后查询。',
                    kind='question', ok=False, rewrite_response=False)
            if decision.route in {RouteKind.WORKFLOW_ACTION, RouteKind.KNOWLEDGE_CHAT}:
                composite = await self._complete_result_delivery(session, None, parsed, text)
                if composite is not None:
                    return composite
            if decision.route in {RouteKind.WORKFLOW_ACTION, RouteKind.CLARIFICATION}:
                if recorded and decision.intent is not Intent.UNKNOWN:
                    pending = dict(session.pending_workflow)
                    restart = await self._superseded_input(session, recorded, before, parsed)
                    if restart is not None:
                        return restart
                    session.pending_workflow = pending
            if decision.route is RouteKind.WORKFLOW_ACTION:
                return await self._dispatch(session, parsed, text)
            if decision.route is RouteKind.CLARIFICATION:
                return await self._handle_clarification(session, decision, parsed, recorded, text=text)
            if decision.route is RouteKind.KNOWLEDGE_CHAT:
                return await self._handle_knowledge_chat(session, decision, parsed, text)
            return await self._handle_out_of_scope(session, decision, parsed, text)
        except WorkflowError as exc:
            return await self._handle_blocked(session, exc, parsed)
        except AgentMCPError as exc:
            return self._fail(session, exc)

    # ------------------------------------------------------- conversation
    #: The entity names that describe the rainfall event.
    RAINFALL_ENTITY_FIELDS = frozenset({"rainfall_mode", "date", "start_datetime",
                                        "end_datetime", "lookback_years", "end_year", "station_count", "station_count_mode"})

    async def _superseded_input(self, session: AgentSession, recorded: dict[str, Any],
                                before: dict[str, Any], parsed=None) -> AgentReply | None:
        """Archive the old task and invalidate dependencies before a correction."""
        changed_place = bool({"administrative_area", "river_name", "longitude", "latitude"} & set(recorded))
        changed_rainfall = bool(self.RAINFALL_ENTITY_FIELDS & set(recorded))
        place_locked = bool(session.watershed_confirmed or session.model_input_package_id
                            or session.hydro_run_id)
        rainfall_locked = bool(session.rainfall_package_id or session.model_input_package_id
                               or session.hydro_input_id or session.hydro_run_id)
        if not ((changed_place and place_locked) or (changed_rainfall and rainfall_locked)):
            return None
        old=AgentSession.model_validate(before)
        try:
            await self.prepare_task_edit(session,"location" if changed_place else "rainfall",previous=old)
        except (WorkflowError,AgentMCPError):
            session.river_name=old.river_name
            session.rainfall_request=old.rainfall_request
            raise
        return None

    async def _handle_restore_task(self,session,parsed,text):
        if re.search(r"revision-|标记|标签|第[一二三四五六七八九十\d]+(?:个|套)?方案|方案\s*[A-Za-z]", text):
            from .historical_tasks import select_revisions, history_answer
            choices = select_revisions(self.store, session.session_id, text)
            if len(choices) != 1:
                answer = history_answer(self.store, session, text)
                return await self._finish(session, **answer, rewrite_response=False,
                                          ledger=EvidenceLedger.from_evidence(answer['facts']), clear_error=False)
            return await self.restore_task_revision(session, choices[0]['revision_id'])
        return await self.restore_task_revision(session)

    async def _complete_result_delivery(self, session, reply, parsed, text):
        """Compose explicitly requested read-only outputs using existing handlers."""
        from .result_delivery import requested_deliverables
        if (not session.hydro_run_id or (reply is not None and reply.answer_source == "saved_task_evidence")
                or parsed.intent not in {Intent.DOWNLOAD_ARTIFACT, Intent.SHOW_RESULT,
                                         Intent.ASK_EXPLANATION, Intent.SHOW_VISUALIZATION}):
            return None
        wanted = requested_deliverables(text, parsed)
        if not wanted.compound:
            return None
        self._validate_result_binding(session, {})
        selected = parsed.model_copy(update={'visualization_kind': wanted.figure_kind})
        parts = {}; completed = {}
        # Render first, so the numerical answer can see actual delivery status
        # rather than treating a missing pre-rendered PNG as missing capability.
        operations = [
            ('figure', wanted.figure, self._handle_show_visualization),
            ('download', wanted.download, self._handle_download_artifact),
            ('result', wanted.result, self._handle_show_result),
            ('provenance', wanted.provenance, self._result_provenance),
        ]
        for name, requested, handler in operations:
            if not requested:
                continue
            try:
                part = await handler(session, selected, text)
            except (WorkflowError, AgentMCPError, IllegalTransition) as exc:
                label = {'figure':'图件', 'download':'下载', 'result':'结果说明', 'provenance':'来源追溯'}[name]
                part = AgentReply(text=f'{label}交付尚未完成：{str(exc).splitlines()[0]}',
                                  session=session, kind='error', ok=False)
            parts[name] = part
            completed[name] = bool(part.ok and not part.audit)
            if name == 'figure':
                completed[name] &= bool(part.images)
            elif name == 'download':
                completed[name] &= bool(part.facts.get('artifact'))
        ordered = [parts[k] for k in ('result', 'figure', 'download', 'provenance') if k in parts]
        facts = {}; images = []; figures = []
        for part in ordered:
            facts.update(part.facts)
            # Each component numbers image_index from zero; merge with offsets.
            figures.extend(v.model_copy(update={'image_index': v.image_index + len(images)})
                           if v.image_index is not None else v for v in part.visualizations)
            images.extend(part.images)
        facts['result_delivery'] = {'run_id': session.hydro_run_id, 'completed': completed}
        self._events(session).append('result_delivery', **facts['result_delivery'])
        return AgentReply(text='\n\n'.join(p.text for p in ordered), session=session,
                          kind='result' if all(completed.values()) else 'question',
                          ok=all(completed.values()), facts=facts, images=images, visualizations=figures,
                          audit=[a for p in ordered for a in p.audit],
                          answer_source=parts.get('result', ordered[0]).answer_source)

    async def _result_provenance(self, session, parsed, text):
        """Report the bound run and saved input sources, without new computation."""
        require_operation(session, Operation.SHOW_RESULT)
        payload = await self._call(session, self.hydro, 'get_hydro_run_result', run_id=session.hydro_run_id)
        result = payload.get('data') or payload
        self._validate_result_binding(session, result)
        rain = session.rainfall_context or {}
        plan = session.last_plan or {}
        start, end = rain.get('event_start'), rain.get('event_end')
        preset = (result.get('safety') or {}).get('parameter_preset') or plan.get('parameter_preset')
        overrides = plan.get('parameter_overrides')
        if overrides is None and session.run_input_signature:
            overrides = dict(session.parameter_overrides)
        lines = [f'本次运行编号：{session.hydro_run_id}。']
        if start and end:
            zone = session.rainfall_request.get('source_timezone') or '未保存时区'
            lines.append(f'降雨输入事件：{start} 至 {end}（{zone}）；这是降雨输入窗口。')
        else:
            lines.append('降雨输入起止时间未保存完整证据，需复核本次输入文件。')
        if preset:
            lines.append(f'参数来源：本次运行采用 {preset} 预设，'
                         + ('覆盖项证据未保存完整。' if overrides is None else
                            '有已保存的参数覆盖项。' if overrides else '未覆盖预设参数。'))
        else:
            lines.append('本次参数来源未保存完整证据，需复核运行配置。')
        facts = {'run_id': session.hydro_run_id, 'rainfall_event': rain,
                 'parameter_preset': preset, 'parameter_overrides': overrides, 'result': result}
        return await self._finish(session, '\n'.join(lines), facts=facts,
                                  kind='result', ok=bool(start and end and preset and overrides is not None),
                                  rewrite_response=False)
    async def _saved_reuse_reply(self, session, text):
        if '复用' not in text:
            return None
        from .task_context import task_fact_reply
        fact = task_fact_reply(text, session)
        if not fact or fact['topic'] != 'task_reuse':
            return None
        return await self._finish(session, fact['text'], kind='explanation',
            facts={'chat_topic':'task_reuse','run_id':session.hydro_run_id},
            answer_source='saved_task_context', rewrite_response=False, clear_error=False)

    async def _handle_clarification(self, session: AgentSession, decision: DialogueDecision,
                                    parsed, recorded: dict[str, Any], text: str = '') -> AgentReply:
        """Ask the one question that matters, and say what was understood."""
        reuse = await self._saved_reuse_reply(session, text)
        if reuse is not None:
            return reuse
        # District candidates remain in pending_workflow until the model resolves
        # its question. Do not announce an uncommitted district as already chosen.
        recorded={key:value for key,value in recorded.items() if key!="administrative_area"}
        clarification = ask_one_question(
            session, missing_fields=list(decision.missing_fields),
            intent=decision.intent, ambiguity=decision.ambiguity,
            recorded=describe_recorded(recorded))
        if "station_count" in decision.missing_fields:
            clarification = Clarification(topic="rainfall", question=
                "虚拟雨站数量需要是1～100的整数；请指定一个数量，或改用按面积自动布站。",
                recorded=describe_recorded(recorded))
        elif decision.ambiguity and (decision.ambiguity.rstrip().endswith(("？", "?"))
                or decision.ambiguity == getattr(self.router, "model_clarification", None)):
            clarification = Clarification(topic=clarification.topic, question=decision.ambiguity,
                                          recorded=describe_recorded(recorded))
        return await self._ask_one_question(session, clarification, parsed=parsed,
                                            recorded=recorded)

    async def _handle_blocked(self, session: AgentSession, error: WorkflowError,
                              parsed=None) -> AgentReply:
        """A refused operation, turned into the question that unblocks it."""
        from .state_machine import WorkflowGateError, pending_action

        # A gate's own ``next_step`` is written for the case where the operation
        # *does* run ("任务会在后台执行…"), which is the wrong thing to say when
        # it did not. What the user needs is the one action the workflow is
        # waiting for, which is what ``pending_action`` states.
        if isinstance(error, WorkflowGateError):
            next_step = pending_action(session.state)
            if parsed is not None and getattr(parsed, "intent", None) is Intent.DOWNLOAD_ARTIFACT:
                # "This conversation has nothing to download" is the first thing
                # to say: the workflow's pending step alone reads as if the
                # request had not been understood.
                next_step = "This session has no files to download yet. " + next_step
        else:
            next_step = getattr(error, "next_step", "") or pending_action(session.state)
        clarification = blocked_question(session, next_step=next_step)
        session.note_error(code=type(error).__name__,
                           message=str(error).splitlines()[0][:300],
                           next_step=next_step)
        return await self._ask_one_question(session, clarification, ok=False,
                                            parsed=parsed)

    async def _ask_one_question(self, session: AgentSession, clarification, *,
                                ok: bool = True, parsed=None,
                                recorded: dict[str, Any] | None = None) -> AgentReply:
        self._events(session).append("clarification_asked", **clarification.log_safe())
        session.last_clarification = clarification.render()
        session.pending_question = clarification.question
        # The route the turn *ended* on. A workflow request the state machine
        # refused became a question, and that is what the user saw — the
        # router's original decision stays in the ``dialogue_route`` event.
        session.last_route = RouteKind.CLARIFICATION.value
        # The values the clarification speaks back to the user (a river, a
        # date, a coordinate) go into the ledger for this turn: they are the
        # user's own input, and a reply that quotes them must be able to
        # account for them — the examples in the wording cannot, which is why
        # they carry no quantities.
        station_bounds = ({"minimum_station_count": 1, "maximum_station_count": 100}
                          if "雨站数量" in clarification.question else {})
        return await self._finish(session, clarification.render(), kind="question", ok=ok,
                                  facts={"clarification_topic": clarification.topic,
                                         "station_count_bounds": station_bounds,
                                         "clarification_examples": clarification.examples,
                                         "clarification_recorded": dict(recorded or {}),
                                         "turn_intent": (parsed.intent.value if parsed is not None
                                                         else "unknown")})

    async def _handle_knowledge_chat(self, session: AgentSession, decision: DialogueDecision,
                                     parsed, text: str) -> AgentReply:
        """Answer a concept, capability or current-result question.

        Explaining the current result is the grounded branch of this channel: it
        goes through the same evidence-bound explainer a "why is it low?"
        follow-up uses, so the numbers can only come from the ledger. A question
        about the *boundaries* of the system — why it produces no risk grade, no
        composite score, no engineering verdict — has no evidence to quote and
        is answered from the knowledge base instead.
        """
        reuse = await self._saved_reuse_reply(session, text)
        if reuse is not None:
            return reuse
        topic = decision.topic or domain_knowledge.match_topic(text)
        from .engineering_knowledge import asks_hypothetical_mechanism
        if asks_hypothetical_mechanism(text):
            assessment = self.store.load_assessment(session.session_id) or {}
            if assessment.get('hydro_run_id') != session.hydro_run_id:
                assessment = {}
            else:
                from .input_binding import input_signature
                if session.run_input_signature and session.run_input_signature != input_signature(session):
                    assessment = {}
            answer = await self.responder.respond_general(question=text, session=session,
                                                          topic="rainfall_mechanism", assessment=assessment)
            return await self._finish_knowledge(session, answer, parsed)
        if session.hydro_run_id and topic in {'current_result', 'task_artifacts', 'task_inputs', 'result_distribution'}:
            return await self._handle_explain(session, parsed, text,
                artifact_question=topic in {'task_artifacts', 'task_inputs'})
        from .task_context import task_fact_reply
        if task_fact_reply(text, session):
            answer = await self.responder.respond_general(question=text, session=session, topic=topic)
            return await self._finish_knowledge(session, answer, parsed)
        assessment = self.store.load_assessment(session.session_id)
        contextual=bool(session.selection_context) and any(cue in text for cue in
            ("这场雨","这次降雨","我选的","这样画","框外","分水岭","当前参数","这个边界"))
        if contextual and not any(cue in text for cue in ("计算结果","安全系数","FS","最低值","失稳比例")):
            answer=await self.responder.respond_general(question=text,session=session,topic=topic)
            return await self._finish_knowledge(session,answer,parsed)
        if (not is_system_question(text, topic)
                and (asks_current_result(text) or (assessment and topic in TERM_TOPICS
                                                   and not asks_general_concept(text)))):
            return await self._handle_explain(session, parsed, text)
        if not is_system_question(text, topic):
            answer = await self.responder.respond_general(question=text, session=session, topic=topic)
            return await self._finish_knowledge(session, answer, parsed)
        answer = self.responder.respond(question=text, session=session,
                                        topic=topic, parsed=parsed,
                                        assessment=assessment)
        self._events(session).append("conversation_reply", **answer.log_safe())
        return await self._finish(session, answer.text, kind="explanation",
                                  answer_source="builtin_knowledge",
                                  facts={"chat_topic": answer.topic,
                                         "turn_intent": parsed.intent.value})

    async def _handle_out_of_scope(self, session: AgentSession, decision: DialogueDecision,
                                   parsed, text: str) -> AgentReply:
        """Say what this agent is for, without pretending to answer."""
        from .state_machine import WorkflowState

        next_step = "" if session.state == WorkflowState.NEW.value else session.pending_action()
        answer = self.responder.out_of_scope(question=text, session=session,
                                             next_step=next_step)
        self._events(session).append("conversation_reply", **answer.log_safe())
        return await self._finish(session, answer.text, kind="message",
                                  facts={"chat_topic": answer.topic,
                                         "turn_intent": parsed.intent.value})

    # --------------------------------------------------------------- spatial
    async def _handle_start_assessment(self, session, parsed, text) -> AgentReply:
        if session.hydro_run_id and not any((parsed.administrative_area, parsed.river_name,
                                             parsed.longitude is not None, parsed.latitude is not None)):
            return await self._handle_show_result(session, parsed, text)
        if (parsed.longitude is not None and parsed.latitude is not None
                and re.search(r"(?:按|用|根据|依据)(?:这个|该|上述|给出的)?(?:坐标|经纬度)(?:加载|定位|打开|选区|绘制)", text)
                and not re.search(r"(?:不要|不用|别|不按|不使用).{0,8}(?:坐标|经纬度)", text)):
            return await self._locate_coordinates(session, parsed.longitude, parsed.latitude)
        if parsed.administrative_area and not parsed.river_name:
            require_operation(session, Operation.RECOMMEND_OUTLETS)
            catalog = await self.administrative_areas(session)
            names = [area["name"] for area in catalog.get("areas", [])]
            requested = parsed.administrative_area
            matched = [name for name in names if requested in {
                name, name.removesuffix("区").removesuffix("县")}]
            if len(matched) != 1:
                # Preserve the task; a spelling suggestion is not a selection.
                previous = session.administrative_area
                if not previous:
                    previous = next((name for turn in reversed(session.conversation)
                                     if turn.role == "user" for name in names
                                     if name in turn.text), None)
                if previous:
                    question = (f"你前面提到的是「{previous}」，这次输入「{requested}」。"
                                f"仍要在{previous}绘制小流域，还是改到另一个区县？")
                else:
                    question = (f"当前重庆行政区底图中没有匹配到「{requested}」。"
                                "请核对区县名称；如果是其他地区，也可以给河名或经纬度定位。")
                return await self._ask_one_question(session,
                    Clarification(topic="place", question=question), parsed=parsed)
            record = await self.study_area_map(session,matched[0])
            map_payload=record["map"]
            reply=await self._finish(session,record["message"],kind="question",
                images=[{"mime_type":"image/png","data":map_payload["image_base64"]}])
            reply.outlet_map={k:v for k,v in map_payload.items() if k!="image_base64"}
            reply.outlet_map.update(selection_mode="drawn_study_area",administrative_area=session.administrative_area,
                                   **session.selection_context)
            return reply
        if parsed.longitude is not None and parsed.latitude is not None:
            return await self._locate_coordinates(session,parsed.longitude,parsed.latitude)
        if parsed.river_name:
            return await self._recommend(session, parsed.river_name)
        center=session.selection_context.get("map_center")
        if center:
            return await self._locate_coordinates(session,center["longitude"],center["latitude"])
        return await self._finish(session, describe_missing_fields(Intent.SELECT_OUTLET, session),
                            kind="question")

    async def _recommend(self, session, river_name: str) -> AgentReply:
        record = await self.outlet_candidates_map(session, river_name)
        map_payload = record.get("map") or {}
        images = []
        if map_payload.get("image_base64"):
            images.append({"mime_type": "image/png", "data": map_payload["image_base64"]})
            if record.get("location_hint"):
                text = (f"本地水系没有匹配到「{river_name}」。已根据模型通用知识加载推测区域，位置尚未核实。"
                        "请核对河段，在地图上逐点绘制闭合评价边界，提交覆盖检查后再确认。")
            else:
                text = f"已按「{river_name}」定位 DEM 地图。请在地图上逐点绘制闭合评价边界，提交覆盖检查后再确认。"
        else:
            text = record.get("message") or "没有找到该河流的底图，请补充所在地区或地图中心经纬度。"
        reply = await self._finish(session, text, kind="question", images=images,
                                   facts={"location_hint": record.get("location_hint")})
        if not images:
            return reply
        reply.outlet_map = {key: value for key, value in map_payload.items()
                            if key in {"bbox_wgs84", "axes_rect", "image_width", "image_height", "note", "crs"}}
        reply.outlet_map.update(river_name=river_name, message=record.get("message"),
                                matched_names=record.get("matched_names") or [],
                                location_hint=record.get("location_hint"),selection_mode="drawn_study_area",
                                administrative_area=None,**session.selection_context)
        return reply

    async def administrative_areas(self, session: AgentSession) -> dict:
        payload=await self._call(session,self.spatial,"list_administrative_areas")
        return payload.get("data") or payload

    async def study_area_map(self, session: AgentSession, administrative_area: str) -> dict:
        if session.review_id or session.model_input_package_id or session.hydro_run_id:
            await self.prepare_task_edit(session,"location")
        else:
            self.checkpoint_task(session,"加载地图前")
        require_operation(session,Operation.RECOMMEND_OUTLETS)
        payload=await self._call(session,self.spatial,"render_administrative_dem",
                                 administrative_area=administrative_area)
        record=payload.get("data") or payload
        session.administrative_area=record["administrative_area"]
        session.selection_mode="drawn_study_area"
        session.river_name=None;session.requested_outlet=None;session.location_hint=None
        self._set_drawing_context(session,record,"district",session.administrative_area)
        session.refresh_summary()
        self._advance(session,WorkflowState.OUTLET_CANDIDATES_READY,"等待绘制评价区域边界")
        self.store.save(session)
        return record

    def _set_drawing_context(self, session, record, mode, label):
        from uuid import uuid4
        session.selection_context={"map_id":uuid4().hex,"location_mode":mode,"location_label":label,
                                   "bbox_wgs84":record["map"]["bbox_wgs84"],
                                   "map_view":{k:v for k,v in record["map"].items() if k in
                                          {"bbox_wgs84","axes_rect","image_width","image_height","display_crs"}}}
        record.update(selection_mode="drawn_study_area",**session.selection_context)
        # A newly loaded map supersedes any unconfirmed boundary.
        session.review_id=None;session.watershed_confirmed=False;session.watershed_qa_passed=None
        session.study_area_coordinates=[];session.study_area_qa={}
        session.study_area_draft={}
        session.pending_question=None;session.last_clarification=None
        session.pending_workflow={}

    async def submit_study_area(self, session: AgentSession, administrative_area: str | None,
                                coordinates: list[list[float]], *, map_id: str | None = None,
                                assessment_mode: str = "regional_screening") -> AgentReply:
        from watershed_exchange.domain_policy import ASSESSMENT_MODES, drawn_modeling_allowed
        if assessment_mode not in ASSESSMENT_MODES:
            raise WorkflowError("Choose a valid assessment mode before submitting.")
        if session.selection_mode!="drawn_study_area" or session.administrative_area!=administrative_area:
            raise WorkflowError("The map area changed. Reload the district map before submitting.")
        context=session.selection_context
        if not context.get("map_id") or (map_id is not None and map_id!=context.get("map_id")) or (not administrative_area and map_id!=context.get("map_id")):
            raise WorkflowError("The map changed. Redraw the boundary on the current map.")
        if context.get("bbox_wgs84"):
            w,s,e,n=context["bbox_wgs84"]
            if any(len(p)!=2 or not w<=p[0]<=e or not s<=p[1]<=n for p in coordinates):
                raise WorkflowError("A vertex is outside the current map. Adjust the map and redraw the boundary.")
        if session.review_id or session.rainfall_package_id or session.hydro_run_id:
            await self.prepare_task_edit(session,"boundary")
        else:
            self.checkpoint_task(session,"提交边界前")
        require_operation(session,Operation.DELINEATE)
        location_arguments=({"location_label":context.get("location_label")} if not administrative_area else {})
        payload=await self._call(session,self.spatial,"create_drawn_study_area",
                                 administrative_area=administrative_area,coordinates=coordinates,
                                 assessment_mode=assessment_mode,**location_arguments)
        record=payload.get("data") or payload
        hydro=(record.get("qa") or {}).get("hydrologic_check") or {}
        if not drawn_modeling_allowed(record.get("qa") or {},assessment_mode):
            # A stale spatial service must not unlock a domain using its old
            # coverage-only QA. Keep the review for diagnosis, never confirm it.
            qa=dict(record.get("qa") or {})
            qa["passed"]=False
            if not hydro:
                qa["errors"]=[*(qa.get("errors") or []),"空间服务尚未提供水文合理性检查，请重启空间服务后重新提交边界。"]
            record={**record,"qa":qa}
        session.review_id=record["review_id"]
        session.assessment_mode=assessment_mode
        session.requested_outlet=None
        session.watershed_confirmed=False
        session.watershed_qa_passed=bool(record["qa"]["passed"])
        session.study_area_coordinates=[list(point) for point in coordinates]
        session.study_area_draft={}
        from .task_context import public_spatial_qa
        session.study_area_qa=public_spatial_qa(record["qa"])
        self._advance(session,WorkflowState.WATERSHED_DELINEATED,"用户已绘制评价范围")
        self._advance(session,WorkflowState.AWAITING_WATERSHED_CONFIRMATION
                      if session.watershed_qa_passed else WorkflowState.WATERSHED_QA_FAILED,
                      "等待人工确认评价范围" if session.watershed_qa_passed else "评价范围覆盖或水文检查未通过")
        images=[await self._call_image(session,self.spatial,"render_process_artifact",
                                      review_id=session.review_id,artifact="watershed")]
        session.record_turn("user",f"在{context.get('location_label') or administrative_area}绘制评价区域并提交覆盖与水文合理性检查。")
        reply=await self._finish(session,describe_review(record, brief=True),kind="question",images=images,facts={"review":record})
        session.record_turn("agent",reply.text)
        session.refresh_summary();self.store.save(session)
        self.checkpoint_task(session,"边界检查完成")
        return reply

    async def outlet_candidates_map(self, session: AgentSession, river_name: str,
                                    max_candidates: int = 0) -> dict[str, Any]:
        """Locate a named river for polygon drawing, without auto delineation."""
        if session.review_id or session.model_input_package_id or session.hydro_run_id:
            await self.prepare_task_edit(session,"location")
        else:
            self.checkpoint_task(session,"加载地图前")
        require_operation(session, Operation.RECOMMEND_OUTLETS)
        session.river_name = river_name
        session.selection_mode="drawn_study_area"
        session.administrative_area=None;session.requested_outlet=None
        session.location_hint = None
        payload = await self._call(session, self.spatial, "render_outlet_candidates",
                                   river_name=river_name, max_candidates=0)
        record = dict(payload.get("data") or payload)
        if not (record.get("map") or {}).get("image_base64"):
            hint = await self.locator.resolve(river_name, session)
            hint = hint.model_copy(update={"region": sanitize_user_text(hint.region),
                                           "description": sanitize_user_text(hint.description)})
            session.location_hint = {**hint.model_dump(), "river_name": river_name,
                                     "source": "general_model_knowledge"}
            self._events(session).append("geographic_location", status=hint.status,
                                          confidence=hint.confidence)
            if hint.status == "resolved":
                payload = await self._call(session, self.spatial, "render_outlet_candidates",
                                          river_name=river_name, max_candidates=0,
                                          longitude=hint.longitude, latitude=hint.latitude)
                record = dict(payload.get("data") or payload)
            else:
                record["message"] = (f"已记录「{river_name}」，本地水系未匹配到该名称。"
                                     + ("模型本轮无法补充可靠位置。" if hint.status == "unavailable"
                                        else "仅凭当前名称还不能可靠区分具体河段。")
                                     + "请补充所在区县、乡镇或附近地标；也可以给出地图中心经纬度。")
            record["location_hint"] = session.location_hint
        # Old services and restored conversations must not revive candidate lists.
        record["candidates"] = []
        session.outlet_candidates = []
        if (record.get("map") or {}).get("image_base64"):
            self._set_drawing_context(session,record,"river",river_name)
            record["message"]="请核对河段，在 DEM 地图上逐点绘制闭合评价范围并提交覆盖检查。"
            self._advance(session, WorkflowState.OUTLET_CANDIDATES_READY, "等待手绘评价边界")
        else:
            session.selection_context={};session.review_id=None;session.watershed_confirmed=False;session.watershed_qa_passed=None
            if session.state!="needs_location":
                self._advance(session,WorkflowState.NEEDS_LOCATION,"等待补充定位信息")
        session.refresh_summary();self.store.save(session)
        return record

    async def location_map(self, session, *, river_name=None, longitude=None, latitude=None):
        if longitude is None and latitude is None:
            return await self.outlet_candidates_map(session,river_name)
        if longitude is None or latitude is None:
            raise ValueError("Enter longitude and latitude for the map centre.")
        if session.review_id or session.model_input_package_id or session.hydro_run_id:
            await self.prepare_task_edit(session,"location")
        else:
            self.checkpoint_task(session,"加载地图前")
        require_operation(session,Operation.RECOMMEND_OUTLETS)
        payload=await self._call(session,self.spatial,"render_outlet_candidates",
                                 river_name=river_name or "坐标定位",max_candidates=0,
                                 longitude=longitude,latitude=latitude)
        record=dict(payload.get("data") or payload)
        if not (record.get("map") or {}).get("image_base64"):
            raise WorkflowError(record.get("message") or "No valid elevation data cover this location.")
        session.river_name=river_name;session.administrative_area=None
        session.selection_mode="drawn_study_area";session.requested_outlet=None;session.location_hint=None
        self._set_drawing_context(session,record,"coordinates","Study area at the selected coordinates")
        session.selection_context["map_center"]={"longitude":longitude,"latitude":latitude}
        record.update(session.selection_context)
        record.update(message="The DEM map is centred at the requested coordinates. Draw a closed boundary and submit it for checks before confirming.",candidates=[])
        self._advance(session,WorkflowState.OUTLET_CANDIDATES_READY,"等待手绘评价边界")
        session.refresh_summary();self.store.save(session)
        return record

    async def _locate_coordinates(self, session, longitude, latitude):
        record=await self.location_map(session,river_name=session.river_name,longitude=longitude,latitude=latitude)
        image=record["map"]
        reply=await self._finish(session,record["message"],kind="question",
            images=[{"mime_type":"image/png","data":image["image_base64"]}])
        reply.outlet_map={k:v for k,v in image.items() if k!="image_base64"}
        reply.outlet_map.update(selection_mode="drawn_study_area",administrative_area=None,**session.selection_context)
        return reply

    async def _handle_select_outlet(self, session, parsed, text) -> AgentReply:
        if parsed.longitude is not None and parsed.latitude is not None:
            return await self._locate_coordinates(session, parsed.longitude, parsed.latitude)
        index = parsed.candidate_index
        if index is None:
            return await self._finish(session, describe_missing_fields(Intent.SELECT_OUTLET, session),
                                kind="question")
        candidates = session.outlet_candidates or []
        if not candidates:
            return await self._finish(session, "Draw a closed boundary on the map, or enter a river name or coordinates to change location.", kind="question")
        if not 1 <= index <= len(candidates):
            return await self._finish(
                session,
                sanitize(f"候选只有 {len(candidates)} 个，请选择 1 到 {len(candidates)}。"),
                kind="question",
                facts={"candidate_count": len(candidates), "minimum_index": 1})
        chosen = candidates[index - 1]
        return await self._locate_coordinates(session, float(chosen["longitude"]),float(chosen["latitude"]))

    async def _handle_confirm(self, session, parsed, text) -> AgentReply:
        # The gate refuses this unless a unique, undecided review is pending.
        from .task_context import boundary_draft_changed
        if boundary_draft_changed(session):
            raise WorkflowError("The boundary draft changed. Submit it for checks before confirming it.")
        require_operation(session, Operation.CONFIRM_WATERSHED)
        if not session.has_pending_watershed():
            raise WorkflowError(
                "Select and check a boundary before confirming it.",
                next_step=f"Next action: {session.pending_action()}",
            )
        if session.watershed_qa_passed is False:
            # The server refuses this too; refusing here gives a usable reason
            # instead of a round trip that cannot succeed.
            raise WorkflowError(
                "The boundary failed the spatial checks and cannot be confirmed.",
                next_step="Choose an outlet on the map and delineate a boundary that passes the checks.",
            )
        payload = await self._call(session, self.spatial, "confirm_watershed",
                                   review_id=session.review_id, confirmed=True,
                                   reviewer_note=_reviewer_note(text))
        record = payload.get("data") or payload
        session.watershed_confirmed = True
        session.reviewer_note = record.get("reviewer_note")
        self._advance(session, WorkflowState.WATERSHED_CONFIRMED, "人工确认流域边界")

        confirmation = describe_confirmation(record, confirmed=True, brief=True)
        # If the rainfall period was already given, carry straight on through the
        # intake chain the plan describes: rainfall, inputs, import, validation, plan.
        if not self._rainfall_is_specified(session):
            return await self._finish(session, confirmation, kind="question")
        return await self._run_intake(session, parsed, text, opening=confirmation, include_rainfall=True)

    async def _run_intake(self, session: AgentSession, parsed, text: str, *,
                          opening: str = "", opening_facts: dict[str, Any] | None = None,
                          include_rainfall: bool = False) -> AgentReply:
        """Run the fixed intake chain, stopping at the first refusal.

        Each step's facts are carried forward, because the reply the user sees
        is the concatenation of every step's text: its ledger has to account for
        every number any of those steps printed, or the audit flags the agent's
        own output as unverifiable and no rewrite of it could ever survive.
        """
        steps: list[str] = [opening] if opening else []
        ledger_facts: dict[str, Any] = {**(opening_facts or {}), "intake_completed": False}
        reused = []
        if include_rainfall and not session.rainfall_package_id and not session.hydro_input_id:
            try:
                reply = await self._build_rainfall(session, parsed)
            except (WorkflowError, AgentMCPError) as exc:
                steps.append(self._fail(session, exc).text)
                return await self._finish(session, "\n\n".join(steps), kind="error",
                                          ok=False, facts=ledger_facts)
            steps.append(reply.text)
            ledger_facts.update(reply.facts)
            if not reply.ok:
                return await self._finish(session, "\n\n".join(steps), kind=reply.kind,
                                          ok=False, facts=ledger_facts)

        reply = None
        for step, completed_field in (
                (self.prepare_model_inputs, "model_input_package_id"),
                (self.import_model_inputs, "hydro_input_id"),
                (self.validate_hydro_input, "hydro_input_validated"),
                (self.plan_hydro_run, "hydro_planned")):
            # Imported inputs need not retain a locally prepared package.
            # A real downstream input id subsumes preparation/import only;
            # validation and human approval still have their own gates.
            imported = bool(session.hydro_input_id)
            if (getattr(session, completed_field) or
                    (imported and completed_field == "model_input_package_id")):
                reused.append(completed_field)
                continue
            try:
                reply = await step(session)
            except (WorkflowError, AgentMCPError) as exc:
                steps.append(self._fail(session, exc).text)
                return await self._finish(session, "\n\n".join(steps), kind="error",
                                          ok=False, facts=ledger_facts)
            steps.append(reply.text)
            ledger_facts.update(reply.facts)
            if not reply.ok:
                return await self._finish(session, "\n\n".join(steps), kind=reply.kind,
                                          ok=False, images=reply.images, facts=ledger_facts)
        ledger_facts["intake_completed"] = True
        if reply is None:
            return await self._finish(session, "The inputs and plan are ready. Review the plan and reply “start run” to submit it.",
                                      kind="question", facts=ledger_facts)
        # Keep the step records as evidence; a user needs the completed outcome,
        # not a concatenated transcript of every internal preparation stage.
        ledger_facts['preparation_steps'] = steps
        ledger_facts['reused_preparation_fields'] = reused
        period = session.rainfall_context
        start, end = period.get('event_start'), period.get('event_end')
        opening = f'已准备 {start} 至 {end} 的降雨与模型输入。' if start and end else '降雨和模型输入已准备好。'
        if period.get('selection_mode') == 'historical_max':
            opening = '已选取历史样本最大日。' + opening
        if re.search(r'复用|哪些已有|哪些.*沿用', text) and session.hydro_input_id:
            opening = ('本轮沿用已导入的模型输入' + ('及已通过的输入验证' if 'hydro_input_validated' in reused else '')
                       + '。运行计划使用当前保存的参数；旧运行结果不能代替更新后的结果。' + opening)
        return await self._finish(session, opening + reply.text, kind="question",
                                  facts=ledger_facts)

    async def _handle_reject(self, session, parsed, text) -> AgentReply:
        require_operation(session, Operation.REJECT_WATERSHED)
        payload = await self._call(session, self.spatial, "confirm_watershed",
                                   review_id=session.review_id, confirmed=False,
                                   reviewer_note=_reviewer_note(text))
        record = payload.get("data") or payload
        session.watershed_confirmed = False
        session.reviewer_note = record.get("reviewer_note")
        self._advance(session, WorkflowState.WATERSHED_REJECTED, "人工拒绝流域边界")
        return await self._finish(session, describe_confirmation(record, confirmed=False, brief=True),
                            kind="question")

    async def _handle_set_rainfall(self, session, parsed, text) -> AgentReply:
        """Build the rainfall package, and continue the intake if it can."""
        if (parsed.station_count is not None or parsed.station_count_mode == "auto") and not self._rainfall_is_specified(session):
            preference = ("虚拟雨站将按选区面积自动确定数量。" if parsed.station_count_mode == "auto"
                          else f"虚拟雨站数量已设为{parsed.station_count}个。")
            return await self._finish(session, preference + "请说明降雨日期、时段，或使用资料期内样本最大日。",
                                      kind="question", facts={"station_count": parsed.station_count})
        if (getattr(self.router,"last_source",None)=="model"
                and not session.watershed_confirmed
                and not any((session.rainfall_package_id,session.model_input_package_id,
                             session.hydro_input_id,session.hydro_run_id))
                and self._rainfall_is_specified(session)):
            request = session.rainfall_request or {}
            event = request.get("date")
            if not event and request.get("start_datetime") and request.get("end_datetime"):
                event = f"{request['start_datetime']} 至 {request['end_datetime']}"
            label = f"降雨日期已记录为{event}。" if event else "已记录历史最大日筛选要求。"
            return await self._finish(session,label + "请先绘制评价边界，提交检查并确认后，再准备降雨与模型输入。",
                                      kind="question",facts={"rainfall_inputs":request})
        reply = await self._build_rainfall(session, parsed)
        if not reply.ok:
            return reply
        if session.watershed_confirmed and session.state == WorkflowState.RAINFALL_READY.value:
            return await self._run_intake(session, parsed, text, opening=reply.text,
                                          opening_facts=reply.facts, include_rainfall=False)
        return reply

    async def _build_rainfall(self, session: AgentSession, parsed) -> AgentReply:
        """The rainfall step alone, without continuing the chain."""
        from .task_context import boundary_draft_changed
        if boundary_draft_changed(session):
            raise WorkflowError("边界草稿已修改，请先提交检查并重新确认，再准备降雨。")
        if session.rainfall_package_id:
            await self.prepare_task_edit(session,"rainfall")
        self.checkpoint_task(session,"准备降雨前")
        require_operation(session, Operation.BUILD_RAINFALL)
        request = self._rainfall_request(session, parsed)
        payload = await self._call(session, self.spatial, "build_virtual_rainfall",
                                   request=request)
        # Record what was actually sent, for the next turn and for the event log.
        session.rainfall_request = request.model_dump(mode="json", exclude_none=True)
        record = payload.get("data") or payload
        if request.mode == "explicit_period":
            from datetime import datetime, timedelta
            start = request.start_datetime or f"{request.date} 00:00:00"
            end = request.end_datetime or f"{request.date} 23:00:00"
            expected_start, expected_end = datetime.fromisoformat(start), datetime.fromisoformat(end)
            expected_count = int((expected_end - expected_start) / timedelta(hours=1)) + 1
            mismatches = []
            for name, expected in (("event_start", expected_start), ("event_end", expected_end)):
                if record.get(name) is not None:
                    try:
                        if datetime.fromisoformat(str(record[name])) != expected:
                            mismatches.append(name)
                    except (TypeError, ValueError):
                        mismatches.append(name)
            if record.get("timestamp_count") is not None and record["timestamp_count"] != expected_count:
                mismatches.append("timestamp_count")
            if mismatches:
                self._events(session).append("rainfall_package_rejected", fields=mismatches,
                    expected={"event_start":start,"event_end":end,"timestamp_count":expected_count},
                    actual={k:record.get(k) for k in ("event_start","event_end","timestamp_count")})
                raise WorkflowError("返回降雨包的起止时间或记录数与当前请求不一致，已停止准备模型输入。",
                                    next_step="请核对降雨事件记录后重新准备。")
        session.rainfall_package_id = record.get("package_id")
        session.rainfall_context={key:record[key] for key in (
            "selection_mode","event_start","event_end","timestamp_count","statistics_years",
            "selected_day_basin_station_mean_total_mm","selection_rule","source_note", "station_selection",
            'existing_grid_count','interpolated_station_count','outside_grid_centre_count','source_selection_policy',
            'relocated_source_count','all_model_stations_inside') if key in record}
        session.rainfall_context["station_count"]=len(record.get("stations") or [])
        self._advance(session, WorkflowState.RAINFALL_READY, "已生成虚拟雨站降雨包")
        self.checkpoint_task(session,"降雨准备完成")
        return await self._finish(session, describe_rainfall_package(record, brief=True), kind="question",
                            facts={"rainfall_package": record})

    def _rainfall_request(self, session: AgentSession, parsed) -> RainfallRequestPayload:
        """Build the request from whatever the session and this turn carry.

        Entity values recorded on an earlier turn win over nothing; this turn's
        values win over the recorded ones, so a correction takes effect.
        """
        if session.pending_workflow.get("rainfall_needs_clarification"):
            raise WorkflowError("新的降雨选择尚未澄清，不能继续使用原降雨准备模型。",
                                next_step="请先明确待用降雨日期、完整时段或历史最大日的筛选条件。")
        stored = dict(session.rainfall_request or {})
        fresh: dict[str, Any] = {}
        if parsed.rainfall_mode:
            fresh["mode"] = parsed.rainfall_mode
        if parsed.date:
            fresh["date"] = parsed.date
        if parsed.start_datetime:
            fresh["start_datetime"] = parsed.start_datetime
        if parsed.end_datetime:
            fresh["end_datetime"] = parsed.end_datetime
        if parsed.lookback_years:
            fresh["lookback_years"] = parsed.lookback_years
        if parsed.end_year:
            fresh["end_year"] = parsed.end_year
        if parsed.station_count_mode == "auto":
            fresh["station_count"] = None
        elif parsed.station_count is not None:
            fresh["station_count"] = parsed.station_count
        if fresh.get("date") and not (fresh.get("start_datetime") or fresh.get("end_datetime")):
            stored.pop("start_datetime", None); stored.pop("end_datetime", None)
        elif ((fresh.get("start_datetime") or fresh.get("end_datetime"))
              and not (fresh.get("date") and fresh.get("start_datetime") and fresh.get("end_datetime")
                       and _is_whole_day(fresh["start_datetime"], fresh["end_datetime"], fresh["date"]))):
            stored.pop("date", None); fresh.pop("date", None)
        merged = {**stored, **fresh}
        options={key:merged[key] for key in ("station_count","source_timezone","end_year")
                 if merged.get(key) is not None}

        mode = merged.get("mode")
        if mode == "historical_max" or (mode is None and not any(merged.get(k) for k in
                                                   ("date", "start_datetime", "end_datetime"))):
            return RainfallRequestPayload(
                review_id=session.review_id, mode="historical_max",
                lookback_years=merged.get("lookback_years") or 30,
                **options,
            )
        if merged.get("date"):
            # "全天" and an explicit window are mutually exclusive in the request
            # the server accepts. A whole-day window *is* the date, so send the
            # date; a narrower window is sent as the window.
            start = merged.get("start_datetime")
            end = merged.get("end_datetime")
            if start and end and not _is_whole_day(start, end, merged["date"]):
                return RainfallRequestPayload(
                    review_id=session.review_id, mode="explicit_period",
                    start_datetime=start, end_datetime=end,
                    **options,
                )
            return RainfallRequestPayload(
                review_id=session.review_id, mode="explicit_period",
                date=merged["date"],
                **options,
            )
        return RainfallRequestPayload(
            review_id=session.review_id, mode="explicit_period",
            start_datetime=merged.get("start_datetime"),
            end_datetime=merged.get("end_datetime"),
            **options,
        )

    # ------------------------------------------------------------- model input
    async def prepare_model_inputs(self, session: AgentSession) -> AgentReply:
        require_operation(session, Operation.PREPARE_MODEL_INPUTS)
        payload = await self._call(session, self.spatial, "prepare_model_inputs",
                                   review_id=session.review_id,
                                   rainfall_package_id=session.rainfall_package_id)
        record = payload.get("data") or payload
        session.model_input_package_id = record.get("package_id")
        # The exported directory is a machine path and stays out of the session;
        # the package travels by id.
        self._advance(session, WorkflowState.MODEL_INPUTS_READY, "已生成模型输入包")
        return await self._finish(session, describe_model_inputs(record, brief=True), kind="question",
                            facts={"model_input": record})

    # ----------------------------------------------------------------- hydro
    async def import_model_inputs(self, session: AgentSession,
                                  package: str | None = None) -> AgentReply:
        """Import the prepared package into the hydro service."""
        require_operation(session, Operation.IMPORT_PACKAGE)
        name = package or session.model_input_package_id
        if not name:
            return await self._finish(session, sanitize("还没有可导入的模型输入包。"), kind="error",
                                ok=False)
        payload = await self._call(session, self.hydro, "import_hydro_input_package",
                                   package=name)
        record = payload.get("data") or payload
        session.hydro_input_id = record.get("input_id")
        session.hydro_input_validated = False
        session.hydro_planned = False
        self._advance(session, WorkflowState.HYDRO_INPUT_IMPORTED, "输入包已导入水文服务")
        return await self._finish(session, describe_import(record, brief=True), kind="question",
                            facts={"hydro_input": record})

    async def validate_hydro_input(self, session: AgentSession) -> AgentReply:
        require_operation(session, Operation.VALIDATE_INPUT)
        payload = await self._call(session, self.hydro, "validate_hydro_input",
                                   input_id=session.hydro_input_id)
        report = payload.get("data") or payload
        session.hydro_input_validated = bool(report.get("ok"))
        if session.hydro_input_validated:
            self._advance(session, WorkflowState.HYDRO_INPUT_VALIDATED, "输入验证通过")
        return await self._finish(session, describe_input_validation(report, brief=True),
                            kind="question" if report.get("ok") else "error",
                            ok=bool(report.get("ok")), facts={"validation": report})

    async def plan_hydro_run(self, session: AgentSession) -> AgentReply:
        from .task_context import boundary_draft_changed
        if boundary_draft_changed(session):
            raise WorkflowError("边界草稿已修改，请先重新检查和确认，再生成运行计划。")
        if session.hydro_run_id:
            await self.prepare_task_edit(session,"parameters")
        require_operation(session, Operation.PLAN_RUN)
        overrides = await self._validated_overrides(session)
        request = HydroRunRequestPayload(input_id=session.hydro_input_id,
                                         mode="build_run_qa",
                                         parameter_preset=session.parameter_preset,
                                         parameter_overrides=overrides)
        payload = await self._call(session, self.hydro, "plan_hydro_run",
                                   request=request)
        plan = payload.get("data") or payload
        # Only the whitelisted part is persisted: the service's plan carries an
        # engine directory, which is a machine path and has no place in a
        # session. The live payload is what gets described to the user.
        session.last_plan = minimise_plan(plan)
        session.run_mode = request.mode
        session.hydro_planned = True
        from .input_binding import input_signature
        session.planned_input_signature = input_signature(session)
        self._advance(session, WorkflowState.HYDRO_PLANNED, "已生成运行计划")
        return await self._finish(session, describe_plan(plan, preset=session.parameter_preset, brief=True),
                            kind="question", facts={"plan": plan})

    async def _validated_overrides(self, session: AgentSession) -> dict[str, float]:
        """Check a requested parameter change against the service's own bounds.

        The service refuses an out-of-range value anyway; checking here means the
        user gets the allowed range instead of a round trip that cannot succeed.
        """
        requested = dict(session.parameter_overrides or {})
        if not requested:
            return {}
        try:
            bounds = await self.hydro.parameter_bounds()
        except AgentMCPError as exc:
            raise WorkflowError(
                "无法读取服务的参数范围，不能确认参数覆盖。",
                next_step=f"{exc.next_step} 也可以先使用参数预设运行。") from exc
        unknown = sorted(set(requested) - set(bounds))
        if unknown:
            raise WorkflowError(
                f"服务不支持这些参数：{unknown}",
                next_step=f"可覆盖的参数：{sorted(bounds)}。",
            )
        out_of_range = {
            key: (value, bounds[key]) for key, value in requested.items()
            if not bounds[key][0] <= float(value) <= bounds[key][1]
        }
        if out_of_range:
            details = "；".join(
                f"{key}={value}，允许范围 {low}–{high}"
                for key, (value, (low, high)) in out_of_range.items())
            raise WorkflowError(f"参数超出允许范围：{details}",
                                next_step="请给出范围内的取值，或改用参数预设。")
        return requested

    async def set_parameter_overrides(self, session: AgentSession,
                                      overrides: dict[str, float]) -> AgentReply:
        """Record a parameter change, stating the range before it is used.

        The plan requires the ranges to be restated before a plan is generated,
        so the caller can confirm what will change.
        """
        if not overrides:
            await self.prepare_task_edit(session,"parameters")
            session.parameter_overrides = {}
            return await self._finish(session, sanitize("已清除参数覆盖，将全部使用预设值。"),
                                kind="message")
        try:
            bounds = await self.hydro.parameter_bounds()
        except AgentMCPError as exc:
            return self._fail(session, exc)
        lines = ["参数覆盖将按下列取值生成计划："]
        rejected: list[str] = []
        for key, value in overrides.items():
            if key not in bounds:
                rejected.append(f"{key} 不是可覆盖的参数")
                continue
            low, high = bounds[key]
            lines.append(f"- {key} = {value}（允许范围 {low}–{high}）")
            if not low <= float(value) <= high:
                rejected.append(f"{key}={value} 超出允许范围 {low}–{high}")
        if rejected:
            return await self._finish(session, sanitize("参数被拒绝：\n- " + "\n- ".join(rejected)),
                                kind="error", ok=False,
                                facts={"requested_overrides": overrides,
                                       "parameter_bounds": bounds})
        await self.prepare_task_edit(session,"parameters")
        session.parameter_overrides = {k: float(v) for k, v in overrides.items()}
        self.store.save(session)
        lines.append(f"参数预设仍为 {session.parameter_preset}。")
        lines.append("确认后回复「开始运行」前的下一步：先生成计划。")
        return await self._finish(session, "\n".join(lines), kind="question",
                            facts={"parameter_overrides": session.parameter_overrides,"parameter_bounds":bounds})

    async def start_hydro_run(self, session: AgentSession) -> AgentReply:
        """Submit the planned run. Returns immediately; the job runs in background."""
        from .task_context import boundary_draft_changed
        if boundary_draft_changed(session):
            raise WorkflowError("存在尚未提交的边界修改，请先提交检查并重新确认边界，再生成计划。")
        self.checkpoint_task(session,"提交计算前")
        require_operation(session, Operation.START_RUN)
        request = HydroRunRequestPayload(
            input_id=session.hydro_input_id, mode="build_run_qa",
            parameter_preset=session.parameter_preset,
            parameter_overrides=dict(session.parameter_overrides or {}))
        payload = await self._call(session, self.hydro, "start_hydro_run",
                                   request=request)
        status = payload.get("data") or payload
        session.hydro_run_id = status.get("run_id")
        from .input_binding import input_signature
        session.run_input_signature = input_signature(session)
        session.hydro_run_status = status.get("status")
        session.run_status_detail=self.public_run_detail(status)
        session.qa_passed = None
        self._advance(session, WorkflowState.HYDRO_RUNNING, "后台任务已提交")
        return await self._finish(session,
                                  describe_run_started(status, brief=True) + self._lifetime_note(),
                                  kind="question", facts={"run": status})

    def _lifetime_note(self) -> str:
        """What this transport means for the job's lifetime, said up front.

        The exit notice already states it, but by then the user has stopped
        reading. Under stdio the service is this process's child, so the job
        dies with it — a user who wants the run to finish deserves to know
        before choosing to start it, not after choosing to leave.
        """
        if self.settings.mcp.hydro.transport == "streamable-http":
            return ""
        return '\n当前任务依赖本机服务，关闭服务会终止未完成的任务，请保持服务运行。'

    async def refresh_run_status(self, session: AgentSession) -> dict[str, Any]:
        """Ask the hydro service for the current state and record it.

        Never trusts the stored status: a resumed session must find out what
        actually happened rather than what it last saw.
        """
        require_operation(session, Operation.CHECK_STATUS)
        payload = await self._call(session, self.hydro, "get_hydro_run_status",
                                   run_id=session.hydro_run_id)
        status = payload.get("data") or payload
        state = str(status.get("status") or "")
        self._validate_result_binding(session, {"status": status})
        session.hydro_run_status = state
        session.run_status_detail=self.public_run_detail(status)
        if "qa_passed" in status:
            session.qa_passed=status["qa_passed"]
        if state == "qa_failed" and session.state in {
                WorkflowState.HYDRO_SUCCEEDED.value,WorkflowState.ASSESSMENT_READY.value}:
            self.checkpoint_task(session,"质量复核未通过前")
            self.store.clear_artifacts(session.session_id,("evidence.json","final_assessment.json"))
            self._advance(session,WorkflowState.HYDRO_QA_FAILED,"质量复核未通过，撤销当前有效评价")
        elif state == "succeeded" and session.state == WorkflowState.HYDRO_QA_FAILED.value:
            self._advance(session,WorkflowState.HYDRO_SUCCEEDED,"重新质量检查通过")
        if state in RUN_STATUS_TO_STATE and session.state == WorkflowState.HYDRO_RUNNING.value:
            self._advance(session, RUN_STATUS_TO_STATE[state], f"任务状态变为 {state}")
        return status

    async def reconcile(self, session: AgentSession) -> AgentReply | None:
        """Re-check a run in progress after a restart.

        Returns a reply describing the outcome when something finished while the
        agent was away, and ``None`` when there is nothing to report.
        """
        if session.state != WorkflowState.HYDRO_RUNNING.value or not session.hydro_run_id:
            return None
        try:
            status = await self.refresh_run_status(session)
        except (WorkflowError, AgentMCPError) as exc:
            return self._fail(session, exc)
        state = str(status.get("status") or "")
        if state in ACTIVE_RUN_STATUSES:
            self.store.save(session)
            return None
        text = ("任务在会话中断期间已结束。\n" if state != "running" else "") + \
            describe_run_status(status, brief=True)
        return await self._finish(session, text, kind="result", facts={"run": status})

    # --------------------------------------------------------------- handlers
    async def _handle_start_run(self, session, parsed, text) -> AgentReply:
        from .dialogue_router import requests_plan_preparation
        preparation = requests_plan_preparation(text)
        continuation = text.strip("。！! ") in {"继续", "下一步", "接着", "然后呢", "go on", "continue"}
        withheld = (awaits_panel_approval(text) or bool(re.search(
            r"(?:暂不|不要|先别|不必).*(?:运行|启动|计算)|(?:批准|确认)后", text)))
        if withheld and not (preparation or continuation):
            return await self._finish(session, "本轮不提交运行。需要开始时，请先核对计划，再明确批准启动。",
                                      kind="question", facts={"pending_user_control": "start_hydro_run"})
        if preparation or continuation or withheld:
            if session.hydro_run_id:
                return await self._handle_check_status(session, parsed, text)
            if not self._rainfall_is_specified(session) and not session.hydro_input_id:
                raise WorkflowError("尚未确定待用降雨事件，不能继续准备模型。",
                                    next_step="请明确降雨日期、时段或历史最大日的筛选条件。")
            return await self._run_intake(session, parsed, text,
                                          include_rainfall=not bool(session.hydro_input_id))
        return await self.start_hydro_run(session)

    async def _handle_check_status(self, session, parsed, text) -> AgentReply:
        status = await self.refresh_run_status(session)
        state = str(status.get("status") or "")
        if state in RUN_STATUS_TO_STATE and state != "succeeded":
            # A finished run that did not pass is explained, not summarised as a result.
            return await self._finish(session, describe_run_status(status, brief=True), kind="result",
                                ok=False, facts={"run": status})
        if state == "succeeded":
            return await self.show_result(session)
        return await self._finish(session, describe_run_status(status, brief=True), kind="message",
                            facts={"run": status})

    async def _handle_cancel(self, session, parsed, text) -> AgentReply:
        require_operation(session, Operation.CANCEL_RUN)
        if awaits_panel_approval(text):
            session.pending_question = "请在面板确认取消当前运行。"
            return await self._finish(session, "已记录取消意向，当前任务仍在运行。请在面板批准取消后执行。",
                                      kind="question", facts={"pending_user_control": "cancel_run"})
        payload = await self._call(session, self.hydro, "cancel_hydro_run",
                                   run_id=session.hydro_run_id)
        status = payload.get("data") or payload
        actual_state = status.get("status")
        if actual_state not in ACTIVE_RUN_STATUSES | set(RUN_STATUS_TO_STATE):
            raise WorkflowError("取消请求返回的状态无法确认，尚不能声称任务已停止。",
                                next_step="请刷新运行状态，确认停止后再修改输入。")
        session.hydro_run_status = status.get("status")
        session.run_status_detail=self.public_run_detail(status)
        session.qa_passed=None
        if actual_state in ACTIVE_RUN_STATUSES:
            return await self._finish(session, "已发送取消请求，但服务仍报告任务在运行或排队。请确认任务结束后再修改输入。",
                                      kind="question", facts={"run": status})
        if session.state == WorkflowState.HYDRO_RUNNING.value:
            self._advance(session, RUN_STATUS_TO_STATE[actual_state], "取消请求后的实际任务状态")
        if actual_state != "cancelled":
            return await self._finish(session, "取消请求返回时任务已结束。" + describe_run_status(status, brief=True),
                                      kind="result", facts={"run": status})
        followup = await self._continue_after_cancellation(session, text)
        if followup is not None:
            return AgentReply(text="当前运行已确认取消。\n\n" + followup.text,
                              session=session, kind=followup.kind, ok=followup.ok,
                              facts={**followup.facts, "cancelled_run": status},
                              images=followup.images, visualizations=followup.visualizations,
                              audit=followup.audit)
        return await self._finish(session, describe_cancellation(status), kind="result",
                            facts={"run": status})

    async def _continue_after_cancellation(self, session, text):
        """A fixed, explicitly requested rainfall edit can follow a confirmed stop."""
        if (not re.search(r"(?:把|将)?(?:降雨|雨日|日期|事件)(?:日期)?\s*(?:改|换|设|调整)", text)
                or re.search(r"如果|假如|会怎样|会怎么样|(?:不要|不用|不必|先别).*?(?:改|换|设|调整)"
                             r"|我(?:再|会|自己).*?(?:改|换|设|调整)", text)):
            return None
        supplied = self.router.rules.parse(text, session)
        fields = ("date", "start_datetime", "end_datetime", "rainfall_mode", "lookback_years", "end_year")
        candidate = ParsedIntent(intent=Intent.SET_RAINFALL_PERIOD,
                                 **{field: getattr(supplied, field) for field in fields})
        if not candidate.date and not (candidate.start_datetime and candidate.end_datetime) and candidate.rainfall_mode != "historical_max":
            return None
        await self.prepare_task_edit(session, "rainfall")
        self._remember_entities(session, candidate)
        return await self._handle_set_rainfall(session, candidate, text)

    async def _handle_show_result(self, session, parsed, text) -> AgentReply:
        # "Result" at this stage means the spatial review, not a hydro run
        # that has not been created. This also makes the no-model fallback
        # explain a failed delineation instead of asking to start calculations.
        if session.review_id and not session.hydro_run_id:
            return await self._describe_current_watershed(session)
        # Asking for the result while a job runs is a status question first; once
        # it has finished — however it finished — the full result is fetched, so
        # a failed run can be explained rather than merely reported.
        if session.state == WorkflowState.HYDRO_RUNNING.value:
            status = await self.refresh_run_status(session)
            if str(status.get("status") or "") in ACTIVE_RUN_STATUSES:
                return await self._finish(session, describe_run_status(status, brief=True), kind="message",
                                    ok=True, facts={"run": status})
        return await self.show_result(session, question=text)

    async def show_result(self, session: AgentSession, *, question: str | None = None) -> AgentReply:
        require_operation(session, Operation.SHOW_RESULT)
        payload = await self._call(session, self.hydro, "get_hydro_run_result",
                                   run_id=session.hydro_run_id)
        result = payload.get("data") or payload
        self._validate_result_binding(session, result)
        status = result.get("status") or {}
        state = str(status.get("status") or session.hydro_run_status or "")
        session.hydro_run_status = state
        quality = result.get("quality") or {}
        session.qa_passed = quality.get("passed")
        if state == "qa_failed" and session.state in {
                WorkflowState.HYDRO_SUCCEEDED.value,WorkflowState.ASSESSMENT_READY.value}:
            self.checkpoint_task(session,"质量复核未通过前")
            self.store.clear_artifacts(session.session_id,("evidence.json","final_assessment.json"))
            self._advance(session,WorkflowState.HYDRO_QA_FAILED,"质量复核未通过，撤销当前有效评价")
        if state in RUN_STATUS_TO_STATE and session.state == WorkflowState.HYDRO_RUNNING.value:
            self._advance(session, RUN_STATUS_TO_STATE[state], f"任务状态变为 {state}")
        # Phase 7 fills in the full assessment; here the result is summarised.
        from .evaluation import build_assessment, describe_result_summary

        assessment = build_assessment(session, result, quality)
        self.store.save_evidence(session.session_id, assessment)
        if assessment.get("assessment_status") == "valid_preliminary":
            if session.state != WorkflowState.ASSESSMENT_READY.value:
                self._advance(session, WorkflowState.ASSESSMENT_READY, "评价已生成")
            self.store.save_assessment(session.session_id, assessment)
        # The model files are part of the outcome, so the conversation learns
        # them here rather than waiting to be asked. A failure to list them does
        # not affect the result being reported.
        model: dict[str, Any] = {}
        try:
            model = await self.refresh_swmm_model(session) or {}
        except AgentMCPError as exc:
            self.logger.warning("SWMM 建模成果清单获取失败：%s", exc)
        try:
            await self.refresh_process_artifacts(session)
        except AgentMCPError as exc:
            self.logger.warning("过程源文件清单获取失败：%s", exc)
        return await self._answer_from_task(session, question or '简要说明本次计算的主要结果和空间分布。',
            assessment, fallback=describe_result_summary(assessment), kind='result',
            facts={'result': result, 'assessment': assessment, 'swmm': model,
                   'process_artifacts': session.process_artifacts})

    async def _answer_from_task(self, session, question, assessment, *, fallback, kind='explanation', facts=None,
                                artifact_question=False):
        from .result_responder import answer_result
        from .task_context import task_background
        catalog = [item.model_dump(mode='json') if hasattr(item, 'model_dump') else dict(item)
                   for item in session.process_artifacts]
        if self.llm is not None and session.hydro_run_id and not catalog:
            try:
                await self.refresh_process_artifacts(session)
                catalog = [item.model_dump(mode='json') if hasattr(item, 'model_dump') else dict(item)
                           for item in session.process_artifacts]
            except AgentMCPError:
                pass
        if self.llm is not None and session.hydro_run_id and 'list_hydro_readable_artifacts' in self.hydro.available_tools:
            try:
                payload = await self._call(session, self.hydro, 'list_hydro_readable_artifacts',
                                           run_id=session.hydro_run_id)
                catalog = (payload.get('data') or {}).get('artifacts') or catalog
            except AgentMCPError:
                pass
        async def read(request):
            payload = await self._call(session, self.hydro, 'inspect_hydro_artifact',
                run_id=session.hydro_run_id, **request.model_dump())
            return payload.get('data') or payload
        answer = await answer_result(llm=self.llm, question=question, assessment=assessment,
            background=task_background(session), catalog=catalog, read_artifact=read, fallback=fallback,
            recent_turns=[{'role': t.role, 'text': t.text} for t in session.recent_conversation(6)],
            artifact_question=artifact_question)
        self._events(session).append('result_reasoning', used_model=answer.used_model,
            reads=answer.reads, fallback_reason=answer.fallback_reason, audit_problems=answer.audit_problems)
        return await self._finish(session, answer.text, kind=kind,
            ok=artifact_question or assessment.get('assessment_status') == 'valid_preliminary',
            ledger=answer.ledger, rewrite_response=False,
            answer_source='grounded_model_result' if answer.used_model else 'mcp_evidence',
            facts=facts or {'assessment': assessment})

    @staticmethod
    def _validate_result_binding(session, result):
        from .input_binding import input_signature
        status = result.get('status') or {}
        returned = {value for value in (result.get('run_id'), status.get('run_id')) if value}
        input_ids = {value for value in (result.get('input_id'), status.get('input_id')) if value}
        signature = getattr(session, 'run_input_signature', None)
        if ((returned and returned != {session.hydro_run_id})
                or (input_ids and session.hydro_input_id and input_ids != {session.hydro_input_id})
                or (signature and signature != input_signature(session))):
            from .mcp_clients.base import MCPProtocolError
            raise MCPProtocolError('服务返回的运行编号或输入版本与当前请求不一致，未采用该结果。',
                                   tool='get_hydro_run_result')

    async def _handle_show_visualization(self, session, parsed, text) -> AgentReply:
        from .visualization_reference import normalize_kind
        kind=normalize_kind(parsed.visualization_kind)
        if kind=='watershed' and session.review_id:
            review_payload=await self._call(session,self.spatial,'get_watershed_review',review_id=session.review_id)
            review=review_payload.get('data') or review_payload
            image=await self._call_image(session,self.spatial,'render_process_artifact',review_id=session.review_id,artifact='watershed')
            return await self._finish(session,'这是当前已保存的评价边界图。\n'+describe_review(review),
                                      kind='result',images=[image],facts={'review':review})
        if not session.hydro_run_id:
            return await self._finish(session,'我理解你想查看建模成果图，但当前还没有提交计算任务。'
                                      '已保存的评价边界可以查看；子汇水区划分图需建模分区阶段完成后生成。',kind='question')
        # Completed-stage plots can be inspected while later stages run.
        if session.state == WorkflowState.HYDRO_RUNNING.value:
            await self.refresh_run_status(session)
        require_operation(session, Operation.SHOW_VISUALIZATION)
        payload = await self._call(session, self.hydro,"list_hydro_visualization_options",run_id=session.hydro_run_id)
        options = list((payload.get("data") or {}).get("options") or [])
        if not kind:
            return await self._finish(session, describe_catalog(options), kind="question",
                                facts={"options": options})
        selected=next((item for item in options if item.get('kind')==kind),None)
        if selected is None:
            return await self._finish(session,'我理解你想查看图件，但当前服务没有这种图件类型。\n'+describe_catalog(options),
                                      kind='question',facts={'options':options})
        if selected.get('available') is False:
            return await self._finish(session, '服务目录显示所请求的图件当前不可用，尚无可交付图件。数值结果需独立核对。',
                                      kind='message', ok=False, facts={'unavailable_figure': selected},
                                      rewrite_response=False)
        if session.state==WorkflowState.HYDRO_RUNNING.value and selected.get('stage_id') not in session.run_status_detail.get('completed_stages',[]):
            return await self._finish(session,'你要看的图件所依赖的阶段还未完成，请等该阶段完成后再查看。\n'
                                      +describe_run_status(session.run_status_detail, brief=True),kind='message',facts={'run':session.run_status_detail})
        return await self.render_visualization(session, kind)

    async def render_visualization(self, session: AgentSession, kind: str) -> AgentReply:
        """Submit a figure, wait briefly, and return the image and its files."""
        payload = await self._call(session, self.hydro, "create_hydro_visualization",
                                   run_id=session.hydro_run_id, kind=kind)
        record = payload.get("data") or payload
        visualization_id = record.get("visualization_id")
        session.visualizations.append({"visualization_id": visualization_id, "kind": kind})
        self.store.save(session)

        status = await self._await_visualization(session, visualization_id)
        if status.get("status") != "succeeded":
            return await self._finish(
                session,
                sanitize(f"图件 {kind} 尚未完成（{status.get('status')}）；"
                         f"可以稍后回复「看{kind}的图」重试。"),
                kind="message", ok=False, facts={"visualization": status})

        figure = await self._collect_figure(session, visualization_id, kind=kind)
        image = await self._call_image(session, self.hydro, "get_hydro_visualization",
                                       visualization_id=visualization_id)
        figure.image_index = 0
        self._remember_figure(session, figure)
        return await self._finish(session, describe_visualization_ready(status, brief=True), kind="result",
                                  images=[image], visualizations=[figure],
                                  facts={"visualization": status,
                                         "artifacts": [item.artifact_id
                                                       for item in figure.artifacts]})

    async def _collect_figure(self, session: AgentSession, visualization_id: str, *,
                              kind: str | None = None) -> VisualizationResult:
        """The figure's real files, as the service describes them publicly.

        A failure here does not lose the figure: the picture is still worth
        showing, and the reply says plainly that the download list is missing
        instead of pretending the figure has no source files.
        """
        try:
            payload = await self._call(session, self.hydro, "list_visualization_artifacts",
                                       visualization_id=visualization_id)
        except AgentMCPError as exc:
            self.logger.warning("图件成果清单获取失败：%s", exc)
            return VisualizationResult(visualization_id=visualization_id, kind=kind,
                                       status="succeeded",
                                       warnings=[f"成果清单获取失败：{exc}"])
        data = payload.get("data") or {}
        if kind and not data.get("kind"):
            data = {**data, "kind": kind}
        try:
            return VisualizationResult.model_validate(data)
        except Exception as exc:  # noqa: BLE001 - a shape the agent cannot read
            self.logger.warning("图件成果清单无法解析：%s", exc)
            return VisualizationResult(visualization_id=visualization_id, kind=kind,
                                       status="succeeded",
                                       warnings=["成果清单格式无法解析；请重试查看该图"])

    @staticmethod
    def _remember_figure(session: AgentSession, figure: VisualizationResult) -> None:
        """Replace the placeholder entry with what the service published."""
        entry = figure.model_dump(mode="json")
        session.visualizations = [
            entry if str(item.get("visualization_id")) == figure.visualization_id else item
            for item in session.visualizations]

    # ------------------------------------------------------- artifact downloads
    def visualizations(self, session: AgentSession) -> list[VisualizationResult]:
        """The figures this conversation holds, for restoring it after a reload."""
        return self.artifacts.session_visualizations(session)

    async def download_artifact(self, session: AgentSession,
                                artifact_id: str) -> ArtifactDownload:
        """Open one file for download, after checking this session may see it."""
        return await self.artifacts.open_download(session, artifact_id)

    async def visualization_image(self, session: AgentSession,
                                  visualization_id: str) -> dict[str, Any]:
        """Fetch one of this session's figures again, for a reloaded page.

        Only figures this session holds can be asked for; the id alone is not
        enough, so a figure id from another conversation is a refusal rather
        than a lookup.
        """
        known = {item.visualization_id for item in self.visualizations(session)}
        if visualization_id not in known:
            raise DownloadError("该图件不属于当前会话", code="not_found", http_status=404)
        return await self._call_image(session, self.hydro, "get_hydro_visualization",
                                      visualization_id=visualization_id)

    async def prepare_artifact_export(self, session: AgentSession, artifact_id: str,
                                      export_format: str) -> ArtifactDescriptor:
        """Derive a viewable file (Excel) from a registered one, and remember it."""
        descriptor = await self.artifacts.prepare_export(session, artifact_id,
                                                         export_format)
        self.store.save(session)
        return descriptor

    async def prepare_swmm_bundle(self, session: AgentSession) -> ArtifactDescriptor:
        """Package this session's SWMM model files, and remember the package."""
        descriptor = await self.artifacts.prepare_swmm_bundle(session)
        self.store.save(session)
        return descriptor

    #: Which of a figure's files is offered when the user did not name one: the
    #: raster the figure was drawn from, then a table, then the model files.
    ROLE_PREFERENCE: tuple[str, ...] = ("source_raster", "source_table",
                                        "native_result", "model_input")

    @staticmethod
    def _descriptor_facts(descriptor: Any) -> dict[str, Any]:
        """A descriptor plus the rounded size the reply text may state."""
        payload = descriptor.model_dump(mode="json") if hasattr(descriptor, "model_dump") \
            else dict(descriptor)
        size = payload.get("size_bytes")
        if isinstance(size, int) and size >= 0:
            payload["size_mb"] = round(size / (1024 * 1024), 1)
        return payload

    def _choose_figure(self, session: AgentSession, parsed) -> VisualizationResult | None:
        """The figure a download request refers to, or ``None``.

        A pasted figure id must be one this session holds; otherwise the most
        recent figure is used, narrowed by the kind the user named when they
        named one. Nothing is inferred from a figure that was never drawn here.
        """
        figures = self.visualizations(session)
        cited = (parsed.identifiers or {}).get("visualization_id")
        if cited:
            return next((item for item in figures if item.visualization_id == cited), None)
        if parsed.visualization_kind:
            from .visualization_reference import normalize_kind
            matching = [item for item in figures if normalize_kind(item.kind) == normalize_kind(parsed.visualization_kind)]
            return matching[-1] if matching else None
        return figures[-1] if figures else None

    def _choose_artifact(self, figure: VisualizationResult,
                         parsed) -> tuple[ArtifactDescriptor | None, bool]:
        """The file to serve from one figure: ``(descriptor, needs_deriving)``.

        A named format wins and always means a file that already exists in that
        format. In particular a CSV request returns the original CSV rather than
        silently manufacturing an Excel workbook. With no format named, the
        primary file is the raster the figure came from.
        """
        wanted = parsed.artifact_format
        cited = (parsed.identifiers or {}).get("artifact_id")
        available = figure.artifacts
        if cited:
            return next((item for item in available if item.artifact_id == cited), None), False
        # Conversion happens only for an explicit Excel request. A request for
        # CSV or a generic table always returns the original CSV bytes.
        if wanted == "xlsx":
            original = next((item for item in available if item.format == "xlsx"), None)
            if original is not None:
                return original, False
            for item in available:
                if "xlsx" in (item.exportable or []):
                    return item, True
            return None, False
        if wanted:
            return next((item for item in available if item.format == wanted), None), False
        for role in self.ROLE_PREFERENCE:
            found = next((item for item in available if item.role == role), None)
            if found is not None:
                return found, False
        return (available[0] if available else None), False

    #: Formats that name a file of the SWMM model itself rather than a figure's
    #: source. Those live in the run's model section, not behind a figure.
    SWMM_FORMATS: frozenset[str] = frozenset({"inp", "rpt", "out", "zip", "model_bundle"})

    @staticmethod
    def _matching_process_artifacts(available, parsed, text):
        """Select from public, ready descriptors; never infer download rights."""
        from .visualization_reference import normalize_kind
        wanted = parsed.artifact_format
        matching = [item for item in available if not wanted or item.format == wanted]
        kind = normalize_kind(parsed.visualization_kind)
        terms = {
            "safety_minimum_factor": ("minimum_safety", "minimum_factor", "minimum_fs", "min_fs", "最小安全", "最小fs"),
            "safety_time_series": ("safety_factor_summary", "safety_time", "fs_timeseries", "安全系数时序"),
            "rainfall_station_series": ("rainfall", "降雨"),
            "swmm_runoff_hydrograph": ("runoff", "hydrograph", "径流"),
            "delineate_subcatchments": ("subcatchment", "子汇水区"),
            "rainfall_cumulative": ("rainfall_cumulative", "cumulative_rain", "累积降雨"),
            "rainfall_station_map": ("rainfall_station_map", "雨站位置", "降雨源分布"),
            "build_model_structure": ("model_structure", "模型结构"),
            "analysis_sensitivity_ranking": ("sensitivity", "敏感性"),
            "analysis_sobol_contribution": ("sobol", "参数贡献"),
            "qa_dashboard": ("qa_dashboard", "质量检查"),
        }.get(kind, (kind,) if kind else None)
        if terms:
            matching = [item for item in matching if any(term in
                ((item.title or "") + " " + (getattr(item, "filename", None) or "")).lower() for term in terms)]
        elif re.search(r'稳定(?:性|度)|安全系数|\bfs\b', text, re.IGNORECASE):
            matching = [item for item in matching if item.stage_id == 'safety' or
                        re.search(r'safety|minimum_fs', item.title or '', re.IGNORECASE)]
        if not wanted and not re.search(r"或|或者|任一", text):
            if re.search(r"表|数据|csv|excel", text, re.IGNORECASE):
                matching = [item for item in matching if item.format in {"csv", "xlsx", "json"}]
            elif re.search(r"图|png|tiff", text, re.IGNORECASE):
                matching = [item for item in matching if item.format in {"png", "geotiff", "tif", "tiff", "svg", "pdf"}]
        return matching

    async def _handle_download_artifact(self, session, parsed, text) -> AgentReply:
        """Serve the file behind a figure, or explain what can be served."""
        require_operation(session, Operation.DOWNLOAD_ARTIFACT)
        if parsed.artifact_format in self.SWMM_FORMATS:
            return await self._handle_swmm_download(session, parsed)

        cited = (parsed.identifiers or {}).get("artifact_id")
        if cited:
            available = self.artifacts.process_artifacts(session)
            if not available:
                available = await self.refresh_process_artifacts(session)
            chosen = next((item for item in available if item.artifact_id == cited), None)
            if chosen is not None:
                return await self._finish(
                    session, describe_artifact_ready(self._descriptor_facts(chosen),
                                                     figure_title="过程源文件"),
                    kind="result", facts={"artifact": self._descriptor_facts(chosen)})

        figure = self._choose_figure(session, parsed)
        if figure is None:
            available = self.artifacts.process_artifacts(session)
            if not available:
                available = await self.refresh_process_artifacts(session)
            cited = (parsed.identifiers or {}).get("artifact_id")
            if cited:
                chosen = next((item for item in available
                               if item.artifact_id == cited), None)
                if chosen is not None:
                    return await self._finish(
                        session, describe_artifact_ready(self._descriptor_facts(chosen),
                                                         figure_title="过程源文件"),
                        kind="result", facts={"artifact": self._descriptor_facts(chosen)})
            if not cited:
                matching = self._matching_process_artifacts(available, parsed, text)
                from .result_delivery import permits_any_file
                if matching and permits_any_file(text):
                    # The user permits one suitable file. Prefer its original
                    # numerical source over an exported duplicate or a log.
                    matching.sort(key=lambda item: (
                        not bool(re.search(r'minimum_safety|minimum_fs|min_fs', item.title or '', re.IGNORECASE)),
                        item.stage_id == 'export',
                        item.role not in {'source_raster', 'source_table'},
                        item.title or '', item.artifact_id))
                    matching = matching[:1]
                if len(matching) == 1:
                    chosen = matching[0]
                    return await self._finish(
                        session, describe_artifact_ready(self._descriptor_facts(chosen),
                                                         figure_title="过程源文件"),
                        kind="result", facts={"artifact": self._descriptor_facts(chosen)})
                if matching:
                    return await self._finish(
                        session, describe_artifact_choices(
                            "过程源文件", [self._descriptor_facts(item) for item in matching]),
                        kind="question", facts={"artifacts": [self._descriptor_facts(item)
                                                               for item in matching]})
            if cited:
                return await self._finish(session, "你指定的成果编号未登记在当前运行中，请核对编号或从当前成果清单选择。",
                                          kind="question", ok=False, rewrite_response=False)
            # No figure yet, but the run's own model files may still be there:
            # "download the results" should reach them rather than stop at the
            # absence of a picture.
            if self.artifacts.swmm_artifacts(session) or parsed.artifact_format:
                model = await self.refresh_swmm_model(session)
                if model.get("artifacts"):
                    return await self._handle_swmm_download(session, parsed)
            return await self._finish(session, describe_no_figure_artifacts(),
                                      kind="question")
        if not figure.artifacts:
            await self._refresh_figure(session, figure)
            figure = self._choose_figure(session, parsed) or figure
        chosen, derive = self._choose_artifact(figure, parsed)
        if chosen is None:
            return await self._finish(
                session,
                describe_artifact_choices(figure.title or figure.visualization_id,
                                          [self._descriptor_facts(item)
                                           for item in figure.artifacts]),
                kind="question", facts={"artifacts": [self._descriptor_facts(item)
                                                      for item in figure.artifacts]})
        others = [item for item in figure.artifacts if item.artifact_id != chosen.artifact_id]
        facts: dict[str, Any] = {"artifact": self._descriptor_facts(chosen),
                                 "figure": figure.visualization_id}
        if derive:
            descriptor = await self.prepare_artifact_export(
                session, chosen.artifact_id, parsed.artifact_format or "xlsx")
            facts["artifact"] = self._descriptor_facts(descriptor)
            facts["derived_from"] = chosen.artifact_id
            return await self._finish(
                session,
                describe_artifact_ready(self._descriptor_facts(descriptor),
                                        figure_title=figure.title, derived=True),
                kind="result", facts=facts)
        return await self._finish(
            session,
            describe_artifact_ready(self._descriptor_facts(chosen),
                                    figure_title=figure.title,
                                    alternatives=[self._descriptor_facts(item)
                                                  for item in others]),
            kind="result", facts=facts)

    async def _refresh_figure(self, session: AgentSession,
                              figure: VisualizationResult) -> None:
        """Re-ask the service for a figure's files, when the session has none.

        A figure rendered before this release, or one whose listing failed the
        first time, can still be completed — without pretending the empty list
        was the answer.
        """
        try:
            refreshed = await self._collect_figure(session, figure.visualization_id,
                                                   kind=figure.kind)
        except AgentMCPError as exc:  # noqa: BLE001 - already described
            self.logger.warning("图件成果清单刷新失败：%s", exc)
            return
        if refreshed.artifacts:
            self._remember_figure(session, refreshed)
            self.store.save(session)

    async def _handle_swmm_download(self, session, parsed) -> AgentReply:
        """Serve one file of the run's SWMM model, or the whole package."""
        if parsed.artifact_format == "model_bundle":
            return await self._handle_swmm_bundle(session)
        wanted = parsed.artifact_format
        available = self.artifacts.swmm_artifacts(session)
        if not available:
            model = await self.refresh_swmm_model(session)
            available = self.artifacts.swmm_artifacts(session)
            if not model:
                return await self._finish(session, describe_no_figure_artifacts(),
                                          kind="question")
        cited = (parsed.identifiers or {}).get("artifact_id")
        if cited:
            chosen = next((item for item in available if item.artifact_id == cited), None)
        elif wanted:
            chosen = next((item for item in available if item.format == wanted), None)
        else:
            chosen = next((item for item in available if item.role == "model_input"), None) \
                or (available[0] if available else None)
        if chosen is None:
            return await self._finish(
                session,
                describe_artifact_choices("SWMM 建模成果",
                                          [self._descriptor_facts(item)
                                           for item in available]),
                kind="question",
                facts={"artifacts": [self._descriptor_facts(item) for item in available]})
        return await self._finish(
            session,
            describe_artifact_ready(
                self._descriptor_facts(chosen), figure_title="SWMM 建模成果",
                alternatives=[self._descriptor_facts(item) for item in available
                              if item.artifact_id != chosen.artifact_id]),
            kind="result", facts={"artifact": self._descriptor_facts(chosen),
                                  "swmm": session.swmm_model or {}})

    async def _handle_swmm_bundle(self, session) -> AgentReply:
        """Prepare (or reuse) the run's SWMM model package."""
        descriptor = await self.prepare_swmm_bundle(session)
        model = session.swmm_model or {}
        complete = bool(model.get("bundle_complete", model.get("complete", True)))
        missing = list(model.get("bundle_missing_required")
                       or model.get("missing_required") or [])
        payload = self._descriptor_facts(descriptor)
        payload["missing_required"] = missing
        return await self._finish(
            session,
            describe_swmm_bundle_ready(payload, complete=complete,
                                       reused=bool(model.get("bundle_reused"))),
            kind="result",
            facts={"artifact": self._descriptor_facts(descriptor),
                   "complete": complete, "missing_required": missing})

    async def refresh_swmm_model(self, session: AgentSession) -> dict[str, Any]:
        """Learn which SWMM model files this run has; public descriptors only."""
        model = await self.artifacts.refresh_swmm_model(session)
        if model:
            self.store.save(session)
        return model

    async def refresh_process_artifacts(self, session: AgentSession) -> list[ArtifactDescriptor]:
        """Learn every original run file; only public descriptors are persisted."""
        artifacts = await self.artifacts.refresh_process_artifacts(session)
        self.store.save(session)
        return artifacts

    async def _await_visualization(self, session: AgentSession,
                                   visualization_id: str) -> dict[str, Any]:
        """Poll a render with backoff, bounded by the configured wait."""
        import asyncio

        polling = self.settings.polling
        waited = 0.0
        interval = polling.initial_interval_s
        status: dict[str, Any] = {}
        while waited < polling.maximum_wait_s:
            payload = await self._call(session, self.hydro,
                                       "get_hydro_visualization_status",
                                       visualization_id=visualization_id)
            status = payload.get("data") or payload
            if str(status.get("status")) in {"succeeded", "failed"}:
                return status
            await asyncio.sleep(interval)
            waited += interval
            interval = min(interval * polling.backoff_factor, polling.maximum_interval_s)
        return status

    async def _handle_explain(self, session, parsed, text, *, artifact_question=False) -> AgentReply:
        """Answer a term or "why" question from this session's own evidence.

        The saved assessment and read-only artifacts belong to this session's
        run. Legacy assessments refresh their statistics from that same run.
        """
        from .evaluation import explain
        from .scientific_policy import EvidenceLedger

        self._validate_result_binding(session, {})
        assessment = self.store.load_assessment(session.session_id)
        if assessment and assessment.get('hydro_run_id') != session.hydro_run_id:
            # Saved results from an older task version are never current facts.
            assessment = None
        if artifact_question and session.hydro_run_id:
            if not assessment or not assessment.get('result_details'):
                payload = await self._call(session, self.hydro, 'get_hydro_run_result', run_id=session.hydro_run_id)
                from .evaluation import build_assessment
                result = payload.get('data') or payload
                self._validate_result_binding(session, result)
                assessment = build_assessment(session, result)
            return await self._answer_from_task(session, text, assessment,
                fallback=explain(session, parsed, text, assessment), artifact_question=True)
        if session.hydro_run_id and (not assessment or not assessment.get('result_details')):
            if session.state in {WorkflowState.HYDRO_SUCCEEDED.value, WorkflowState.ASSESSMENT_READY.value}:
                return await self.show_result(session, question=text)
        if assessment:
            return await self._answer_from_task(session, text, assessment,
                fallback=explain(session, parsed, text, assessment))
        if not assessment and session.review_id and asks_current_result(text):
            return await self._describe_current_watershed(session)
        ledger = EvidenceLedger.from_assessment(assessment)
        return await self._finish(session, explain(session, parsed, text, assessment),
                                  kind="explanation", ledger=ledger,
                                  facts={"assessment_status": (assessment or {}).get(
                                      "assessment_status")})

    async def _describe_current_watershed(self, session) -> AgentReply:
        payload = await self._call(session, self.spatial, "get_watershed_review",
                                   review_id=session.review_id)
        record = payload.get("data") or payload
        intro = ("当前只有流域划分结果，还没有稳定性计算结果。"
                 + ("空间质量检查通过，仍需核对目标河段与边界。" if (record.get("qa") or {}).get("passed")
                    else "这次划分未通过空间质量检查，不能作为目标流域使用。"))
        return await self._finish(session, intro + "\n" + describe_review(record),
                                  kind="explanation", facts={"review": record})

    async def _handle_unknown(self, session, parsed, text) -> AgentReply:
        return await self._finish(session, describe_unknown(session, text), kind="question")

    # ------------------------------------------------------------------ misc
    async def _call_image(self, session: AgentSession, client, method: str,
                          **arguments) -> dict[str, Any]:
        events = self._events(session)
        events.append("tool_call", tool=method, arguments=_log_safe(arguments))
        payload = await getattr(client, method)(**arguments)
        events.append("tool_result", tool=method, ok=True, status="image")
        return payload

    async def describe_service(self) -> AgentReply:
        session = self.store.create()
        payload = await self._call(session, self.hydro, "get_hydro_capabilities")
        return await self._finish(session, describe_capabilities(payload), kind="message",
                            facts={"capabilities": payload.get("data") or {}})

    async def scope_statement(self) -> str:
        return sanitize(f"{describe_scope()}\n{DISCLAIMER_ZH}")


def _is_whole_day(start: str, end: str, day: str) -> bool:
    """True when a window is exactly midnight-to-23:00 of the given day."""
    try:
        start_day, start_time = str(start).split()
        end_day, end_time = str(end).split()
    except ValueError:
        return False
    return (start_day == day and end_day == day
            and start_time in {"00:00", "00:00:00"}
            and end_time in {"23:00", "23:00:00"})


#: Values safe to state to the language model: names, numbers and units. The
#: model never sees a path, an identifier, or an MCP response.
def _canonical_facts(ledger: EvidenceLedger) -> list[Fact]:
    facts: list[Fact] = []
    for entry in ledger.entries:
        if entry.value is None:
            continue
        facts.append(Fact(name=entry.name, value=f"{entry.value:g}",
                          unit=entry.unit, source=entry.source))
    return facts


def _reviewer_note(text: str) -> str:
    """A note that satisfies the server, derived from what the user actually said."""
    note = str(text).strip()
    return note[:2000] if note else "用户通过对话确认"


def _log_safe(arguments: dict[str, Any]) -> dict[str, Any]:
    """Event-log view of tool arguments: ids and scalars, nothing bulky."""
    from pydantic import BaseModel

    safe: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, dict):
            safe[key] = {k: v for k, v in value.items()
                         if isinstance(v, (str, int, float, bool)) or v is None}
        else:
            safe[key] = f"<{type(value).__name__}>"
    return safe
