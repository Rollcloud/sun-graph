from starlette.responses import FileResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import settings, views

static = StaticFiles(directory=str(settings.STATIC_DIR))


def favicon(request):
    return FileResponse(str(settings.STATIC_DIR / "images" / "favicon.ico"))


routes = [
    Route("/", views.home, name="home"),
    Route("/favicon.ico", favicon, name="favicon"),
    Mount("/static", static, name="static"),
]
