import pytest
from backend.tools.web_tools import search_web, read_url_content

def test_search_web(mocker):
    mock_ddgs = mocker.MagicMock()
    mock_ddgs.__enter__.return_value.text.return_value = [{"title": "Test", "body": "Test body", "href": "http://test"}]
    mocker.patch("backend.tools.web_tools.DDGS", return_value=mock_ddgs)
    
    res = search_web("test query")
    assert len(res) == 1
    assert res[0]["title"] == "Test"

def test_read_url_content(mocker):
    mock_resp = mocker.MagicMock()
    mock_resp.content = b"<html><body><h1>Test Page</h1><p>Content</p><script>alert('no')</script></body></html>"
    mocker.patch("requests.get", return_value=mock_resp)
    
    text = read_url_content("http://dummy")
    assert "Test Page Content" in text
    assert "alert" not in text

def test_web_tools_exceptions(mocker):
    mocker.patch("requests.get", side_effect=Exception("Timeout"))
    mocker.patch("backend.tools.web_tools.DDGS", side_effect=Exception("DDGS error"))
    
    from backend.tools.web_tools import search_web, read_url_content
    assert "Search failed" in search_web("test")[0]["error"]
    assert "Error reading" in read_url_content("http://test.com")
