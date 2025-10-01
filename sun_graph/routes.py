from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import settings, views

static = StaticFiles(directory=str(settings.STATIC_DIR))

routes = [
    Route("/", views.home, name="home"),
    Mount("/static", static, name="static"),
]
