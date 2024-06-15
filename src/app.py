import gradio as gr
from image_preprocessing import ImagePreprocessor
from feature_extraction import FeatureExtractor
from nearest_neighbors import NearestNeighborsSearch


class ImageSearchApp:
    
    def __init__(self, data_path="data/test_data"):
        # ... (загрузка данных, создание индекса)

    def search(self, image, class_name=None):
        # ... (предобработка, извлечение признаков, поиск)
        # ... (возврат результатов для отображения в Gradio)

iface = ImageSearchApp()
gr.Interface(
    fn=iface.search,
    inputs=[
        gr.Image(type="filepath"),
        gr.Dropdown(choices=["bags", "full_body", ...], label="Класс (необязательно)")
    ],
    outputs=[gr.Gallery(label="Похожие изображения")],
    title="Поиск похожих изображений",
).launch()