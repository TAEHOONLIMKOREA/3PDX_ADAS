from fastapi import FastAPI
from pydantic import BaseModel

# FastAPI 객체 생성
app = FastAPI()

# 루트 경로에 GET 요청 시 응답
@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}

# 파라미터 받기
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}

# 요청 바디를 위한 모델 정의
class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True

@app.post("/items/")
def create_item(item: Item):
    return {"message": "Item created", "item": item}