import os
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from data_io import read_exr, write_exr
from opengl_util import create_compute_shader, create_compute_program, create_texture, create_window, read_texture
import OpenGL.GL as gl

def fgsr_I_init(shader_sources):
     # 初始化opengl和创建着色器
    programs = []
    create_window(1280, 720, "fgsr_I")
    for source in shader_sources:
        with open(f"shader/{source}", "r") as f:
            shader_source = f.read()
        shader = create_compute_shader(shader_source)
        program = create_compute_program(shader)
        programs.append(program)
    return programs

def fgsr_I(mv_0, mv_1, depth_0, depth_1, color_0, color_1,programs):
    height, width = color_1.shape[0], color_1.shape[1]
    # 生成查询对象
    queries = glGenQueries(len(programs))

    total_time = 0
    # Step1:MV's splatting
    # 使用传入的ndarray中的数据创建纹理
    # mv1:1->a mv2:0->a
    mv_tex = create_texture(mv_1, width, height)
    mv_1_tex = create_texture(mv_0, width, height)
    depth_1_tex = create_texture(depth_1, width, height)
    color_1_tex = create_texture(color_1, width, height)
    
    # 创建存放结果的纹理
    warp_mv1_tex = create_texture(None, width, height)
    warp_mv2_tex = create_texture(None, width, height)
    warp_depth_tex = create_texture(None, width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT)

    query = queries[0]
    glUseProgram(programs[0])
    # 将纹理绑定到着色器
    in_textures = [mv_tex, mv_1_tex, depth_1_tex, color_1_tex]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [warp_mv1_tex,warp_mv2_tex, warp_depth_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)

    gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)

    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    gl.glEndQuery(gl.GL_TIME_ELAPSED)


    warp_mv1 = read_texture(warp_mv1_tex, width, height)
    warp_mv2 = read_texture(warp_mv2_tex, width, height)
    warp_depth = read_texture(warp_depth_tex, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT)
    warp_depth = ((2147483647 - np.expand_dims(warp_depth, axis=-1)) / 65535).astype(np.float32)
    warp_mv1 = np.reshape(warp_mv1, (height, width, 4))
    warp_mv2 = np.reshape(warp_mv2, (height, width, 4))
    warp_depth = np.reshape(warp_depth, (height, width, 1))
    # Step2:refine MV
    mv_tex1 = warp_mv1_tex

    # 创建存放结果的纹理
    inpaint_mv1_tex = create_texture(None, width, height)

    query = queries[1]
    glUseProgram(programs[1])
    # 将纹理绑定到着色器
    in_textures = [mv_tex1]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [inpaint_mv1_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)

    gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)

    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    gl.glEndQuery(gl.GL_TIME_ELAPSED)
    # 从结果纹理中读取数据
    inpaint_mv1 = read_texture(inpaint_mv1_tex, width, height)
    inpaint_mv1 = np.reshape(inpaint_mv1, (height, width, 4))

    mv_tex2 = warp_mv2_tex

    # 创建存放结果的纹理
    inpaint_mv2_tex = create_texture(None, width, height)

    query = queries[1]
    glUseProgram(programs[1])
    # 将纹理绑定到着色器
    in_textures = [mv_tex2]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [inpaint_mv2_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)

    gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)

    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    gl.glEndQuery(gl.GL_TIME_ELAPSED)
    # 从结果纹理中读取数据
    inpaint_mv2 = read_texture(inpaint_mv2_tex, width, height)
    inpaint_mv2 = np.reshape(inpaint_mv2, (height, width, 4))

    #Step3:warp/SR

    depth_0_tex = create_texture(depth_0, width, height)
    color_0_tex = create_texture(color_0,width,height)
    # 创建存放结果的纹理
    warp_color1_tex = create_texture(None, width, height)
    warp_depth1_tex = create_texture(None, width, height)
    warp_color2_tex = create_texture(None, width, height)
    warp_depth2_tex = create_texture(None, width, height)


    query = queries[2]
    glUseProgram(programs[2])
    # 将纹理绑定到着色器
    in_textures = [inpaint_mv1_tex,inpaint_mv2_tex,color_1_tex,color_0_tex,depth_1_tex,depth_0_tex]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [warp_color1_tex,warp_depth1_tex,warp_color2_tex,warp_depth2_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)

    gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)

    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    gl.glEndQuery(gl.GL_TIME_ELAPSED)
    # 从结果纹理中读取数据
    warp_color1 = read_texture(warp_color1_tex, width, height)
    warp_color1 = np.reshape(warp_color1, (height, width, 4))
    warp_color2 = read_texture(warp_color2_tex, width, height)
    warp_color2 = np.reshape(warp_color2, (height, width, 4))
    warp_depth1 = read_texture(warp_depth1_tex, width, height)
    warp_depth1 = np.reshape(warp_depth1, (height, width, 4))
    warp_depth2 = read_texture(warp_depth2_tex, width, height)
    warp_depth2 = np.reshape(warp_depth2, (height, width, 4))

    # Step4: pixel selection & blending
    # 创建存放结果的纹理
    predict_color_tex = create_texture(None, width, height)

    query = queries[3]
    glUseProgram(programs[3])
    # 将纹理绑定到着色器
    in_textures = [warp_color1_tex,warp_color2_tex,warp_depth1_tex,warp_depth2_tex]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [predict_color_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)

    gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)

    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    gl.glEndQuery(gl.GL_TIME_ELAPSED)
    # 从结果纹理中读取数据
    pred_color = read_texture(predict_color_tex, width, height)
    pred_color = np.reshape(pred_color, (height, width, 4))
    
    return warp_color1,warp_color2,warp_depth1,warp_depth2,pred_color

def fgsr_I_main(label_index, label_path, seq_path, save_path, scene_name, programs, debug=False):
    input_index = (label_index + 1) // 2
    print(scene_name)
    mv_0 = read_exr(os.path.join(seq_path, f"{scene_name}MotionVector.{str(input_index-1).zfill(4)}.exr"), channel=4)
    mv_1 = read_exr(os.path.join(seq_path, f"{scene_name}MotionVector.{str(input_index).zfill(4)}.exr"), channel=4)
    depth_0 = read_exr(os.path.join(label_path, f"{scene_name}SceneDepth.{str(label_index-1).zfill(4)}.exr"), channel=1)
    depth_1 = read_exr(os.path.join(label_path, f"{scene_name}SceneDepth.{str(label_index+1).zfill(4)}.exr"), channel=1)
    color_0 = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index-1).zfill(4)}.exr"), channel=4)
    color_1 = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index+1).zfill(4)}.exr"), channel=4)
    depth_0 = np.repeat(depth_0, 4, axis=-1)
    depth_0[...,3] = 1
    depth_1 = np.repeat(depth_1, 4, axis=-1)
    depth_1[...,3] = 1
    
    warp_mv1,warp_mv2,wpd1,wpd2,pc= fgsr_I(mv_0, mv_1, depth_0, depth_1, color_0, color_1,programs)
    
    #write_exr(os.path.join(save_path, f"{scene_name}WarpColor.{str(label_index).zfill(4)}.exr"), warp_color)
    write_exr(os.path.join(save_path, f"{scene_name}WarpColor1.{str(label_index).zfill(4)}.exr"), warp_mv1)
    write_exr(os.path.join(save_path, f"{scene_name}WarpColor2.{str(label_index).zfill(4)}.exr"), warp_mv2)
    # write_exr(os.path.join(save_path, f"{scene_name}WarpDepth.{str(i+1).zfill(4)}.exr"), warp_depth)
    write_exr(os.path.join(save_path, f"{scene_name}WarpColor.{str(label_index).zfill(4)}.exr"), pc)
    if debug:
        # 读取并保存label
        color_gt = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index).zfill(4)}.exr"), channel=4)
        # mv_gt = read_exr(os.path.join(label_path, f"{scene_name}MotionVector.{str(label_index).zfill(4)}.exr"), channel=4)
        write_exr(os.path.join(save_path, f"{scene_name}GTColor.{str(label_index).zfill(4)}.exr"), color_gt)
        write_exr(os.path.join(save_path, f"{scene_name}D0.{str(label_index).zfill(4)}.exr"), depth_0)
        write_exr(os.path.join(save_path, f"{scene_name}D1.{str(label_index).zfill(4)}.exr"), depth_1)
        write_exr(os.path.join(save_path, f"{scene_name}WarpDepth1.{str(label_index).zfill(4)}.exr"), wpd1)
        write_exr(os.path.join(save_path, f"{scene_name}WarpDepth2.{str(label_index).zfill(4)}.exr"), wpd2)
        # write_exr(os.path.join(save_path, f"{scene_name}FillMotionVector.{str(label_index).zfill(4)}.exr"), fill_mv)