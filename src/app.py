import gradio as gr  # Импорт gradio для создания веб-интерфейса
import os

from image_preprocessing import ImagePreprocessor
from feature_extraction import FeatureExtractor
from nearest_neighbors import NearestNeighborsSearch


class ImageSearchApp:
    """
    Класс для создания веб-приложения с использованием Gradio.
    """

    def __init__(self, data_path="data/test_data", model_name="resnet18", embedding_dim=512):
        """
        Инициализация ImageSearchApp.

        Args:
            data_path (str): Путь к папке с данными, разделенными по классам.
            model_name (str): Название модели для извлечения признаков (например, "resnet18").
            embedding_dim (int): Размерность эмбеддинга, получаемого из модели.
        """
        self.data_path = data_path
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # Создание экземпляров классов для предобработки, 
        # извлечения признаков и поиска ближайших соседей
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor(self.model_name)
        self.nn_search = NearestNeighborsSearch(self.embedding_dim)

        # Загрузка данных и создание индекса
        self.load_data_and_create_index()

    def load_data_and_create_index(self):
        """
        Загружает данные из data_path, извлекает признаки 
        и строит индекс для поиска ближайших соседей.
        """
        for class_name in os.listdir(self.data_path):
            class_path = os.path.join(self.data_path, class_name)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)
                    # Загрузка и предобработка изображения
                    image = self.preprocessor.preprocess(image_path)
                    # Извлечение признаков
                    embedding = self.feature_extractor.extract_features(image)
                    # Добавление эмбеддинга в индекс
                    self.nn_search.add_embeddings(embedding)

    def search(self, image_path, class_name=None, top_k=5):
        """
        Осуществляет поиск похожих изображений и возвращает результаты 
        в формате, подходящем для отображения в Gradio.

        Args:
            image_path (str): Путь к изображению для поиска.
            class_name (str, optional): Название класса для поиска.
            top_k (int, optional): Количество возвращаемых похожих изображений.

        Returns:
            list: Список путей к найденным изображениям, 
                  подходящий для отображения в Gradio Gallery.
        """
        # Загрузка и предобработка изображения
        image = self.preprocessor.preprocess(image_path)
        # Извлечение признаков
        query_embedding = self.feature_extractor.extract_features(image)
        # Поиск ближайших соседей
        neighbors_indices = self.nn_search.search(query_embedding, top_k)

        # !!! Здесь нужно реализовать логику для:
        # 1. Фильтрации найденных изображений по class_name (если он указан)
        # 2. Преобразования neighbors_indices в список путей к изображениям

        # Возврат путей к найденным изображениям (пример)
        found_image_paths = [f"path/to/image_{i}.jpg" for i in neighbors_indices] 
        return found_image_paths

# Создание экземпляра приложения
iface = ImageSearchApp()

# Создание интерфейса Gradio
gr.Interface(
    fn=iface.search,  # Функция, вызываемая при нажатии на кнопку "Submit"
    inputs=[
        gr.Image(type="filepath", label="Загрузите изображение"),  # Вход для изображения
        gr.Dropdown(choices=["bags", "full_body", ...], label="Класс (необязательно)"),  # Выбор класса 
        gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Количество результатов"), # Выбор количества результатов
    ],
    outputs=[gr.Gallery(label="Похожие изображения")],  # Вывод результатов в виде галереи
    title="Поиск похожих изображений",  # Заголовок приложения
).launch()