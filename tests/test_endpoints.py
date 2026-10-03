import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# You can add more tests for your endpoints below, for example:
#
# @pytest.mark.asyncio
# async def test_upload_document():
#     ...
