## Text Splitter

A simple Python library for searching images in particular database

Obviously not cherry-picked exmaple below
### Input:
<img src="images_thumb/image.png" alt="drawing" width="100"/>

### Output:
<img src="images_thumb/image-1.png" alt="drawing" width="100"/>
<img src="images_thumb/image-2.png" alt="drawing" width="100"/>

### Using Gradio


![alt text](images_thumb/image-3.png)

### Installation

1. Install [poetry](https://python-poetry.org/docs/#installation)

2. Use git clone
```bash 
git clone https://github.com/Eddudos/image_search.git
# https://gitfront.io/r/user-8370390/RM8DUsMeE6vv/image-search.git
cd image_search
```
3. Install dependencies and env
```bash
poetry install
```
4. Enjoy

### Usage

```
python src/cli.py search --image-path "<path-to-query-image>" --class-name "<class-name>" --top_k <k-nearest-neighbors>
```

### Exmaple

```bash
python src/cli.py search --image-path "data/test_data/bags/item-3afd9c7a-28e8-456b-9449-64aede6d4745.jpg" --class-name "bags" --top_k 6

Output:
Indices of nearest embeddings: [ 32 173  91 180 167  66] 

data/test_data/bags/item-3afd9c7a-28e8-456b-9449-64aede6d4745.jpg
data/test_data/bags/item-eac279ff-1dd2-41e9-a763-cb378ca027ce.jpg
data/test_data/bags/item-df8a8676-1d90-4be2-a33b-ebb69cb95875.jpg
data/test_data/bags/item-ce61a6cc-eeb6-4373-802f-6e1f21c0dbc2.jpg
data/test_data/bags/item-fefd308f-a01e-4867-b3b2-ebf371e696eb.jpg
data/test_data/bags/item-8df1e025-80a6-4d00-b110-a7a02d0a2fa4.jpg
```

### Contributing

Thank me very much. 

### License

[MIT](https://choosealicense.com/licenses/mit/)
