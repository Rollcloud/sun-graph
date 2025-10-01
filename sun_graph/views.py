from starlette.responses import PlainTextResponse


async def home(request):
    return PlainTextResponse("Hello, world!")
