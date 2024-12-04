from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


# 创建一个opengl窗口
def create_window(width, height, title):
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
    glutInitWindowSize(width, height)
    glutCreateWindow(title)


# 创建并编译计算着色器
def create_compute_shader(shader_source):
    shader = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(shader, shader_source)
    glCompileShader(shader)
    compile_status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not compile_status:
        info_log = glGetShaderInfoLog(shader)
        raise Exception(f"Compute shader compilation failed: {info_log}")
    return shader


# 创建并链接计算着色器程序
def create_compute_program(shader):
    program = glCreateProgram()
    glAttachShader(program, shader)
    glLinkProgram(program)
    link_status = glGetProgramiv(program, GL_LINK_STATUS)
    if not link_status:
        info_log = glGetProgramInfoLog(program)
        raise Exception(f"Compute shader program linking failed: {info_log}")
    return program
    

# 创建纹理
def create_texture(image, width, height, internal_format=GL_RGBA32F, format=GL_RGBA, type=GL_FLOAT):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, width, height)
    if image is not None:
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, image)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


# 用glReadPixels读取纹理数据
def read_texture(texture, width, height, format=GL_RGBA, type=GL_FLOAT):
    framebuffer = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    image = glReadPixels(0, 0, width, height, format, type)
    glFinish()
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDeleteFramebuffers(1, [framebuffer])
    return image

def create_background_buffer_textures(images,width, height, num_levels=4):
    textures = glGenTextures(num_levels)  # 创建 4 个纹理对象
    current_width, current_height = width, height
    
    for i, texture in enumerate(textures):
        glBindTexture(GL_TEXTURE_2D, texture)  # 绑定当前纹理
        
        # 为纹理分配存储空间（每层大小逐渐减小）
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, current_width, current_height)
        
        # 设置纹理参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        if images is not None:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, current_width,  current_height, GL_RGBA, GL_FLOAT, images[i])
        glBindTexture(GL_TEXTURE_2D, 0)
        # 更新下一层的分辨率 (每次减半)
        current_width = max(1, current_width // 2)
        current_height = max(1, current_height // 2)
    
    glBindTexture(GL_TEXTURE_2D, 0)  # 解绑纹理
    return textures  # 返回纹理对象 ID 列表
