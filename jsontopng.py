# -*- coding: utf-8 -*-
import base64
import json
import os
import os.path as osp
import imgviz
import PIL.Image
from labelme.logger import logger
from labelme import utils
import numpy as np

def main():
    # 打印警告信息
    # logger.warning("This script is aimed to demonstrate how to convert the JSON file to a single image dataset.")
    # logger.warning("It won't handle multiple JSON files to generate a real-use dataset.")

    json_dirname = "tttt"  # 输入JSON文件目录
    out_dir_base = "tttt1"  # 输出文件夹路径

    # 查找目录下所有的json文件
    json_files = []
    for root, dirs, files in os.walk(json_dirname):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    for json_file in json_files:
        # 确定输出文件夹路径
        out_dir = out_dir_base
        if not osp.exists(out_dir):
            os.makedirs(out_dir)

        # 读取JSON文件内容
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        imageData = data.get("imageData")

        # 如果JSON文件中包含图像数据，则直接读取图像数据
        # 如果不包含图像数据,则寻找该目录下的图片
        if not imageData:
            imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)

        # 将标签名称映射到数值
        label_name_to_value = {"_background_": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        # 生成标签图像
        lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

        # 将标签图像变为纯白的二值图像
        lbl_binary = np.where(lbl > 0, 255, 0).astype(np.uint8)

        # 生成标签可视化图像
        lbl_viz = imgviz.label2rgb(lbl, imgviz.asgray(img), label_names=None, loc="rb")

        # 保存图像和标签
        # PIL.Image.fromarray(img).save(osp.join(out_dir, "img.png"))
        # np.save(osp.join(out_dir, "label.npy"), lbl)  # 保存标签图像为.npy格式
        # PIL.Image.fromarray(lbl_binary).save(osp.join(out_dir, "label_binary.png"))  # 保存二值标签图像为PNG格式
        # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))
        PIL.Image.fromarray(lbl_binary).save(osp.join(out_dir, osp.basename(json_file).replace(".json", ".png")))

        # 保存标签名称到文本文件
        with open(osp.join(out_dir, "label_names.txt"), "w") as f:
            f.write(
                "img.png: Original image \n"
                "label.npy: Semantic map in .npy format \n"
                "label_binary.png: Binary map with pixel values of 0 and 255 \n"
                "label_viz.png: Overlay the semantic map and the original map \n" +
                osp.basename(json_file).replace(".json", ".jpg") + ": The pixel value is a binary image of the label sequence number\n\n"
            )
            for name, value in label_name_to_value.items():
                f.write("class: " + name + "\t\tvalue: " + str(value) + "\n")


        logger.info("Saved to: {}".format(out_dir))
        print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()