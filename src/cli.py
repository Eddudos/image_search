import fire  # Для создания CLI
import os
import numpy as np

from image_preprocessing import ImagePreprocessor
from feature_extraction import FeatureExtractor
from nearest_neighbors import NearestNeighborsSearch

class ImageSearchCLI:
    """
    Класс для реализации CLI приложения поиска похожих изображений.
    """

    def __init__(self, data_path="data/test_data", model_name="resnet18", embedding_dim=512):
        """
        Инициализация ImageSearchCLI.

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

        concatenated_embeddings = np.empty((0, 512))

        for class_name in os.listdir(self.data_path):
            class_path = os.path.join(self.data_path, class_name)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)
                    # Загрузка и предобработка изображения
                    image = self.preprocessor.preprocess(image_path)
                    # Извлечение признаков
                    embedding = self.feature_extractor.extract_features(image)
                    embedding = np.expand_dims(embedding, axis=0)
                    # Добавление эмбеддинга в индекс
                    concatenated_embeddings = np.concatenate((concatenated_embeddings, embedding), axis=0)
        self.nn_search.add_embeddings(concatenated_embeddings)

    def search(self, image_path, class_name=None, top_k=6):
        """
        Осуществляет поиск похожих изображений.

        Args:
            image_path (str): Путь к изображению для поиска.
            class_name (str, optional): Название класса для поиска.
            top_k (int, optional): Количество возвращаемых похожих изображений.

        Returns:
            list: Список путей к найденным изображениям.
        """
        # Загрузка и предобработка изображения
        image = self.preprocessor.preprocess(image_path)
        # Извлечение признаков
        query_embedding = self.feature_extractor.extract_features(image)
        # Поиск ближайших соседей
        neighbors_indices = self.nn_search.search(query_embedding, top_k)

        print(neighbors_indices)

        # !!! Здесь нужно реализовать логику для:
        # 1. Фильтрации найденных изображений по class_name (если он указан)
        # 2. Преобразования neighbors_indices в список путей к изображениям

        # Возврат путей к найденным изображениям
        return found_image_paths

if __name__ == "__main__":
    fire.Fire(ImageSearchCLI)