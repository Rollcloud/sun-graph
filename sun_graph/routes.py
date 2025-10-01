from starlette.routing import Route

from . import views

routes = [
    Route("/", views.home, name="home"),
]
