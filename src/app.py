import gradio as gr

from cli import ImageSearchCLI

# Создание экземпляра приложения
iface = ImageSearchCLI()

# Создание интерфейса Gradio
gr.Interface(
    fn=iface.search, 
    inputs=[
        gr.Image(type="filepath", label="Загрузите изображение"), 
        gr.Dropdown(choices=["bags", "full_body", "glasses", "lower_body", "other", "shoes", "upper_body"], label="Класс (необязательно)"),  
        gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Количество результатов"),
    ],
    outputs=[gr.Gallery(label="Похожие изображения")],  
    title="Поиск похожих изображений", 
).launch()