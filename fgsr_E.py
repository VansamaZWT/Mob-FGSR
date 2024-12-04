import os
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from data_io import read_exr, write_exr
from opengl_util import create_compute_shader, create_compute_program, create_texture, create_window, read_texture
import OpenGL.GL as gl
import time

def fgsr_E_init(shader_sources):
    # 初始化opengl和创建着色器
    programs = []
    create_window(1280, 720, "fgsr_E")
    for source in shader_sources:
        with open(f"shader/{source}", "r") as f:
            shader_source = f.read()
        shader = create_compute_shader(shader_source)
        program = create_compute_program(shader)
        programs.append(program)
    return programs

def fgsr_E(mv_0, mv_1, depth_0, depth_1, color_0, color_1,programs):
    height, width = color_1.shape[0], color_1.shape[1]
    # 生成查询对象
    queries = glGenQueries(len(programs))

    total_time = 0
    # Step1:MV's splatting
    # 使用传入的ndarray中的数据创建纹理
    mv_tex = create_texture(mv_1, width, height)
    mv_1_tex = create_texture(mv_0, width, height)
    depth_tex = create_texture(depth_1, width, height)
    color_tex = create_texture(color_1, width, height)
    
    # 创建存放结果的纹理
    warp_mv_tex = create_texture(None, width, height)
    warp_depth_tex = create_texture(None, width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT)

    query = queries[0]
    glUseProgram(programs[0])
    # 将纹理绑定到着色器
    in_textures = [mv_tex, mv_1_tex, depth_tex, color_tex]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [warp_mv_tex, warp_depth_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)

    gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)

    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    gl.glEndQuery(gl.GL_TIME_ELAPSED)


    warp_mv = read_texture(warp_mv_tex, width, height)
    warp_depth = read_texture(warp_depth_tex, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT)
    warp_depth = ((2147483647 - np.expand_dims(warp_depth, axis=-1)) / 65535).astype(np.float32)
    warp_mv = np.reshape(warp_mv, (height, width, 4))
    warp_depth = np.reshape(warp_depth, (height, width, 1))
    # Step2:refine MV
    mv_tex1 = warp_mv_tex

    # 创建存放结果的纹理
    inpaint_mv_tex = create_texture(None, width, height)

    query = queries[1]
    glUseProgram(programs[1])
    # 将纹理绑定到着色器
    in_textures = [mv_tex1]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [inpaint_mv_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)

    gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)

    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    gl.glEndQuery(gl.GL_TIME_ELAPSED)
    # 从结果纹理中读取数据
    inpaint_mv = read_texture(inpaint_mv_tex, width, height)
    inpaint_mv = np.reshape(inpaint_mv, (height, width, 4))

    #Step3:disocclusions filling
    disc_mv_tex = inpaint_mv_tex

    # 创建存放结果的纹理
    fill_mv_tex = create_texture(None, width, height)

    query = queries[2]
    glUseProgram(programs[2])
    # 将纹理绑定到着色器
    in_textures = [disc_mv_tex,mv_tex]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [fill_mv_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)

    gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)

    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    gl.glEndQuery(gl.GL_TIME_ELAPSED)
    # 从结果纹理中读取数据
    fill_mv = read_texture(fill_mv_tex, width, height)
    fill_mv = np.reshape(fill_mv, (height, width, 4))

    #Step4:warp/SR

    # 创建存放结果的纹理
    warp_color_tex = create_texture(None, width, height)

    query = queries[3]
    glUseProgram(programs[3])
    # 将纹理绑定到着色器
    in_textures = [fill_mv_tex,color_tex]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [warp_color_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)

    gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)

    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    gl.glEndQuery(gl.GL_TIME_ELAPSED)
    # 从结果纹理中读取数据
    warp_color = read_texture(warp_color_tex, width, height)
    warp_color = np.reshape(warp_color, (height, width, 4))



    delete_textures = [mv_tex, mv_1_tex, depth_tex, color_tex, warp_mv_tex, warp_depth_tex,inpaint_mv_tex,disc_mv_tex,fill_mv_tex]
    glDeleteTextures(len(delete_textures), delete_textures)
    glFinish()

    for query in queries:
        while gl.glGetQueryObjectiv(query, gl.GL_QUERY_RESULT_AVAILABLE) != gl.GL_TRUE:
            pass

        # 获取查询结果（时间，单位是纳秒）
        time_elapsed = gl.glGetQueryObjectiv(query, gl.GL_QUERY_RESULT)
        # 累加每个 shader 的执行时间
        total_time += time_elapsed

    # 将总时间从纳秒转换为毫秒
    total_time_in_ms = total_time / 1e6
    print(f"Total execution time: {total_time_in_ms:.4f} ms")

    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL Error: {error}")

    return warp_mv,inpaint_mv,fill_mv,warp_color

def fgsr_E_main(label_index, label_path, seq_path, save_path, scene_name, programs, debug=False):
    input_index = (label_index - 1) // 2
    print(scene_name)
    mv_0 = read_exr(os.path.join(seq_path, f"{scene_name}MotionVector.{str(input_index-1).zfill(4)}.exr"), channel=4)
    mv_1 = read_exr(os.path.join(seq_path, f"{scene_name}MotionVector.{str(input_index).zfill(4)}.exr"), channel=4)
    depth_0 = read_exr(os.path.join(label_path, f"{scene_name}SceneDepth.{str(label_index-3).zfill(4)}.exr"), channel=4)
    depth_1 = read_exr(os.path.join(label_path, f"{scene_name}SceneDepth.{str(label_index-1).zfill(4)}.exr"), channel=4)
    color_0 = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index-3).zfill(4)}.exr"), channel=4)
    color_1 = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index-1).zfill(4)}.exr"), channel=4)
    depth_0 = np.repeat(depth_0, 4, axis=-1)
    depth_1 = np.repeat(depth_1, 4, axis=-1)
    
    warp_mv,inpaint_mv,fill_mv,c= fgsr_E(mv_0, mv_1, depth_0, depth_1, color_0, color_1,programs)
    
    #write_exr(os.path.join(save_path, f"{scene_name}WarpColor.{str(label_index).zfill(4)}.exr"), warp_color)
    write_exr(os.path.join(save_path, f"{scene_name}WarpColor.{str(label_index).zfill(4)}.exr"), c)
    # write_exr(os.path.join(save_path, f"{scene_name}WarpDepth.{str(i+1).zfill(4)}.exr"), warp_depth)

    if debug:
        # 读取并保存label
        color_gt = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index).zfill(4)}.exr"), channel=4)
        mv_gt = read_exr(os.path.join(label_path, f"{scene_name}MotionVector.{str(label_index).zfill(4)}.exr"), channel=4)
        write_exr(os.path.join(save_path, f"{scene_name}PreTonemapHDRColor.{str(label_index).zfill(4)}.exr"), color_gt)
        write_exr(os.path.join(save_path, f"{scene_name}MotionVector.{str(label_index).zfill(4)}.exr"), mv_gt)
        write_exr(os.path.join(save_path, f"{scene_name}WarpMotionVector.{str(label_index).zfill(4)}.exr"), warp_mv)
        write_exr(os.path.join(save_path, f"{scene_name}InpaintMotionVector.{str(label_index).zfill(4)}.exr"), inpaint_mv)
        write_exr(os.path.join(save_path, f"{scene_name}FillMotionVector.{str(label_index).zfill(4)}.exr"), fill_mv)
