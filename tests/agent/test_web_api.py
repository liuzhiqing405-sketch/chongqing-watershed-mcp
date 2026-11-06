"""The browser adapter must preserve the agent's session and safety boundary."""

from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from slope_evaluation_agent.agent import AgentReply
from slope_evaluation_agent.config import AgentSettings
from slope_evaluation_agent.session_store import SessionStore
from slope_evaluation_agent.product_cleanup import ProductCleanupMixin
from slope_evaluation_agent.web import create_app


PNG = base64.b64encode(
    b"\x89PNG\r\n\x1a\n" + b"web-adapter-test"
).decode("ascii")


class FakeAgent(ProductCleanupMixin):
    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings
        self.store = SessionStore(settings.sessions)
        self.opened = False
        self.closed = False

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True

    def new_session(self, *, language: str = "zh"):
        return self.store.create(language=language)

    async def handle(self, session, text: str) -> AgentReply:
        session.record_turn("user", text)
        answer = f"已收到：{text}"
        session.record_turn("agent", answer)
        self.store.save(session)
        return AgentReply(
            text=answer,
            session=session,
            facts={"private_path": r"D:\never-expose"},
            images=[{"mime_type": "image/png", "data": PNG}],
        )

    async def reconcile(self, session):
        return None

    async def scope_statement(self) -> str:
        return "区域尺度初步筛查；不是工程安全鉴定。"


def settings_for(tmp_path) -> AgentSettings:
    return AgentSettings.model_validate({
        "sessions": {"root": str(tmp_path / "sessions")},
        "llm": {
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-flash",
        },
    })


def test_static_ui_health_and_lifecycle(tmp_path):
    settings = settings_for(tmp_path)
    agent = FakeAgent(settings)
    app = create_app(settings=settings, agent=agent)

    with TestClient(app) as client:
        assert agent.opened is True
        page = client.get("/")
        assert page.status_code == 200
        assert "坡面稳定性评价智能体" in page.text

        health = client.get("/api/v1/health").json()
        assert health["ok"] is True
        assert health["session_product_cleanup"] is True
        assert health["llm"]["provider"] == "deepseek"
        assert health["llm"]["configured"] is False
        assert "API Key" in health["llm"]["reason"]
        assert "api_key" not in str(health).lower()

        scope = client.get("/api/v1/scope")
        assert scope.status_code == 200
        assert "不是工程安全鉴定" in scope.json()["text"]

    assert agent.closed is True


def test_session_message_resume_and_image_are_ui_safe(tmp_path):
    settings = settings_for(tmp_path)
    app = create_app(settings=settings, agent=FakeAgent(settings))

    with TestClient(app) as client:
        created = client.post("/api/v1/sessions", json={"language": "zh"})
        assert created.status_code == 201
        session_id = created.json()["session_id"]

        sent = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"text": "分析大宁河"},
        )
        assert sent.status_code == 200
        payload = sent.json()
        assert payload["reply"]["text"] == "已收到：分析大宁河"
        assert "facts" not in payload["reply"]
        assert r"D:\never-expose" not in sent.text
        assert payload["reply"]["audit_passed"] is True

        image_url = payload["reply"]["images"][0]["url"]
        image = client.get(image_url)
        assert image.status_code == 200
        assert image.content.startswith(b"\x89PNG")

        resumed = client.get(f"/api/v1/sessions/{session_id}").json()
        assert [turn["role"] for turn in resumed["conversation"]] == ["user", "agent"]
        assert resumed["conversation"][0]["text"] == "分析大宁河"


def test_invalid_ids_and_blank_messages_are_refused(tmp_path):
    settings = settings_for(tmp_path)
    app = create_app(settings=settings, agent=FakeAgent(settings))

    with TestClient(app) as client:
        assert client.get("/api/v1/sessions/not-a-session").status_code == 404
        assert client.get(
            "/api/v1/sessions/not-a-session/images/anything.png"
        ).status_code == 404

        created = client.post("/api/v1/sessions", json={}).json()
        response = client.post(
            f"/api/v1/sessions/{created['session_id']}/messages",
            json={"text": "   "},
        )
        assert response.status_code == 422


def test_history_is_listed_restored_and_deleted_with_its_images(tmp_path):
    settings = settings_for(tmp_path)
    agent = FakeAgent(settings)
    app = create_app(settings=settings, agent=agent)

    with TestClient(app) as client:
        first = client.post("/api/v1/sessions", json={}).json()
        second = client.post("/api/v1/sessions", json={}).json()
        sent = client.post(
            f"/api/v1/sessions/{first['session_id']}/messages",
            json={"text": "分析大宁河"},
        ).json()
        image_url = sent["reply"]["images"][0]["url"]
        assert client.get(image_url).status_code == 200

        history = client.get("/api/v1/sessions?limit=100")
        assert history.status_code == 200
        rows = history.json()["sessions"]
        assert [row["session_id"] for row in rows][0] == first["session_id"]
        first_row = next(row for row in rows if row["session_id"] == first["session_id"])
        assert first_row["title"] == "分析大宁河"
        assert first_row["message_count"] == 2

        deleted = client.delete(f"/api/v1/sessions/{first['session_id']}")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": first["session_id"],
                                  "cleanup": {"deleted_ids": [], "retained_ids": []}}
        assert client.get(f"/api/v1/sessions/{first['session_id']}").status_code == 404
        assert client.get(image_url).status_code == 404
        assert client.get(f"/api/v1/sessions/{second['session_id']}").status_code == 200


def test_an_active_hydro_session_must_be_cancelled_before_deletion(tmp_path):
    settings = settings_for(tmp_path)
    agent = FakeAgent(settings)
    app = create_app(settings=settings, agent=agent)

    with TestClient(app) as client:
        created = client.post("/api/v1/sessions", json={}).json()
        session = agent.store.load(created["session_id"])
        session.state = "hydro_running"
        session.hydro_run_status = "running"
        agent.store.save(session)

        response = client.delete(f"/api/v1/sessions/{session.session_id}")

        assert response.status_code == 409
        assert "先取消任务" in response.json()["detail"]
        assert agent.store.exists(session.session_id) is True


def test_history_limits_and_invalid_deletes_are_checked(tmp_path):
    settings = settings_for(tmp_path)
    app = create_app(settings=settings, agent=FakeAgent(settings))

    with TestClient(app) as client:
        assert client.get("/api/v1/sessions?limit=0").status_code == 422
        assert client.get("/api/v1/sessions?limit=201").status_code == 422
        assert client.delete("/api/v1/sessions/not-a-session").status_code == 404
        assert client.delete("/api/v1/sessions/session-ffffffffffff").status_code == 404
