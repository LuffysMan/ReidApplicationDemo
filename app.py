import PIL.Image as Image
import os
import time

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


from src.reid_model import ReidModel
from helper import parse_image_path_and_copy, clean_folder


app = Flask(__name__)


@app.route('/<int:max_amount>', methods=['POST', 'GET'])                            # 添加路由
def index(max_amount):
    if request.method == 'POST':
        f = request.files['file']
 
        basepath = os.path.dirname(__file__)                        # 当前文件所在路径
        path_to_static_images = os.path.join(basepath, "static/images")
        clean_folder(path_to_static_images)
        
        upload_path = os.path.join(path_to_static_images, secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
 
        # 转换一下图片格式和名称
        img = Image.open(upload_path).convert('RGB')
        img.save(os.path.join(path_to_static_images, 'query.jpg'))

        # 查找相似图片, 返回图片列表
        reid_model = ReidModel()
        src_image_paths = reid_model.get_similar_image_rank_list(img, max_length=max_amount)

        #将图片拷贝到static/images目录下; 并解析图片信息
        image_info_list = parse_image_path_and_copy(src_image_paths, path_to_static_images)
        return render_template('upload_ok.html',image_info_list = image_info_list, val1=time.time())
 
    return render_template('upload.html')


if __name__ == "__main__":
    app.run("localhost", 5000, debug=True)
