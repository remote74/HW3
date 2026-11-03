from fastapi import APIRouter

from service import dataset as dataset_service

router = APIRouter()

@router.get("/dataset/preview")
def get_dataset_preview(n: int = 5):
    return dataset_service.convert_to_dataset_rows(n)


@router.get("/dataset/info")
def get_dataset_info():
    return dataset_service.get_dataset_info()

@router.get("/dataset/split-info")
def get_dataset_split_info():
    return dataset_service.get_split_info()
