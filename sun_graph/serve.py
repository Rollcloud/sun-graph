from starlette.applications import Starlette

from .routes import routes

app = Starlette(routes=routes)
