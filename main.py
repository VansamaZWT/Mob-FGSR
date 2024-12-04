import os
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from data_io import read_exr, write_exr, read_matrix
from opengl_util import create_compute_shader, create_compute_program, create_texture, create_window, read_texture
from fgsr_E import fgsr_E_main,fgsr_E_init

def main(root="E:/workspace/zwtdataset/", sub_paths=["Bunker/train1-30fps-combine"], mode="fgE", debug=False):
    if mode == "fgE":
        shader_sources = ["fgsr_me.comp", "fgsr_inpaint.comp", "disocclusion_fill.comp", "warp.comp"]

        programs =fgsr_E_init(shader_sources)
        save_path_ = "./fgsr_E"

    else:
        raise ValueError(f"Invalid mode: {mode}")

    

    # 获取数据元信息
    input_paths = []
    label_paths = []
    scene_names = []
    index_ranges = []
    for sub_path in sub_paths:
        input_path = os.path.join(root, sub_path)
        input_paths.append(input_path)
        scene_names.append(sub_path.split('/')[0])
        # input_path下所有文件名都是xxx.index.exr的格式，找到index的范围
        buffer_names = sorted(os.listdir(input_path))
        index_start = int(buffer_names[0].split('.')[1])
        index_end = int(buffer_names[-1].split('.')[1])
        
        label_path = input_path.replace("3", "6")
        # 确认label_path下的文件的index范围能够与input_path对齐
        buffer_names = sorted(os.listdir(label_path))
        index_start_label = int(buffer_names[0].split('.')[1]) // 2
        index_end_label = int(buffer_names[-1].split('.')[1]) // 2
        label_paths.append(label_path)
        
        index_ranges.append([max(index_start, index_start_label), min(index_end, index_end_label)])
    
    for scene_name, input_path, index_range, label_path in zip(scene_names, input_paths, index_ranges, label_paths):
        save_path = os.path.join(save_path_, scene_name)
        save_path = os.path.join(save_path, label_path.split('/')[-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for i in range(index_range[0] + 1, index_range[1] + 1):
            label_index = 2 * i + 1
            fgsr_E_main(label_index, label_path, input_path, save_path, scene_name, programs, debug)
            
            print(label_path + ": " + str(label_index))

            # 如果处于debug模式，只循环3次
            if debug:
                if i >= index_range[0] + 5:
                    break


if __name__ == "__main__":
    root = "D:/datasetZWT/me_test"
    sub_paths = ["Bunker/test10-30FPS"]
    main(root, sub_paths, mode="fgE", debug=True)