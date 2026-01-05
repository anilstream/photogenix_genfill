# Standard library
import logging
import tempfile
import time

# Third-party
import uvicorn
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import HttpUrl

# Local application
from genfill_model import FluxOneRewardOutpainter
from genfill_utils import fetch_image_data

flux_outpainter = FluxOneRewardOutpainter()

templates = Jinja2Templates(directory="templates")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(process)d - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
app = FastAPI(
    title="Genfill API",
    version="1.0.0",
    openapi_url="/genfill/openapi.json",
    docs_url="/genfill/docs",
    redoc_url="/genfill/redoc",
)

@app.get('/genfill/status')
def genfill_preset_status_get(request: Request):
    return {'status': 'OK'}


@app.get("/genfill/generate")
def genfill_preset_predict_get(request: Request):
    return templates.TemplateResponse("upload_image.html", {"request": request})


@app.post("/genfill/generate")
async def genfill_preset_predict_post(request: Request, image: UploadFile = File(...), image_url: HttpUrl = Form(None),
                                      height: int = Form(None), width: int = Form(None),
                                      top: int = Form(None), bottom: int = Form(None), left: int = Form(None), right: int = Form(None)):
    try:
        t1 = time.perf_counter()
        logger.info(f"top: {top}, bottom: {bottom}, left: {left}, right: {right}, width: {width}, height: {height}")

        image = fetch_image_data(image_url) if image_url  else await image.read()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_image:

            temp_image.write(image)
            temp_image.flush()

            output = flux_outpainter.run(temp_image.name, top=top,bottom=bottom, left=left, right=right)

        t2 = time.perf_counter() - t1
        logger.info(f"time taken: {t2}")

        return Response(content=output, media_type="image/jpg",
                        headers={'Content-Disposition': f'attachment; filename=processed.jpg'})

    except Exception as e:
        logger.exception(str(e), exc_info=True)
        return {"code": -1, "status": f"prediction failed - {str(e)}"}


if __name__ == "__main__":
    uvicorn.run("genfill_api:app", host="0.0.0.0", port=5007, workers=1)
