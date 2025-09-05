# flash_grid.py
import sys
import glfw
import numpy as np
from OpenGL import GL as gl

VERT_SRC = """
#version 120
attribute vec2 position;
void main(){
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAG_SRC = """
#version 120
uniform vec2 u_cell_px;      // セルのピクセルサイズ（例: 80,80）
uniform float u_line_px;     // 線の太さ（px）
uniform float u_gamma;       // 簡易ガンマ
uniform int u_show_grid;     // 1で表示、0で黒
uniform vec2 u_viewport;     // 画面サイズ（px）

// ちょい簡易なハッシュでカラフル化
float hash21(vec2 p){
    p = fract(p*vec2(0.1031, 0.11369));
    p += dot(p, p.yx+19.19);
    return fract(p.x*p.y);
}

void main(){
    if(u_show_grid == 0){ gl_FragColor = vec4(0,0,0,1); return; }

    vec2 frag = gl_FragCoord.xy;              // 画面上のピクセル座標
    vec2 cell = u_cell_px;
    vec2 m = mod(frag, cell);                  // セル内位置
    float d = min(min(m.x, cell.x - m.x), min(m.y, cell.y - m.y)); // 罫線までの距離
    float line = smoothstep(u_line_px+0.5, u_line_px-0.5, d);      // 線マスク（アンチエイリアス）

    // セルのインデックスから色を作る
    vec2 idx = floor(frag / cell);
    float h = hash21(idx);
    // HSV→RGBもどき（速さ優先）
    float r = abs(h*6.0 - 3.0) - 1.0;
    float g = 2.0 - abs(h*6.0 - 2.0);
    float b = 2.0 - abs(h*6.0 - 4.0);
    vec3 base = clamp(vec3(r,g,b), 0.0, 1.0);

    // 背景は黒、線はカラフル
    vec3 rgb = mix(vec3(0.0), base, line);

    // 目よりカメラ優先：簡易ガンマ補正で白飛び抑制
    rgb = pow(rgb, vec3(1.0/u_gamma));
    gl_FragColor = vec4(rgb, 1.0);
}
"""

def main():
    # パラメータ（お好みで）
    MONITOR_INDEX = 1         # 使うモニタ（0 = プライマリ）
    CELL_PX = (80, 80)         # グリッドのセル（px）
    LINE_PX = 4.0              # 線の太さ（px）
    ON_FRAMES = 1              # 何フレーム点灯するか（例: 240Hzで約8.3ms）
    OFF_FRAMES = 30           # 何フレーム消灯するか（例: 0.5秒）
    CYCLES = 1000              # 何回点滅を繰り返すか
    GAMMA = 2.2                # 簡易ガンマ

    if not glfw.init():
        print("glfw init failed"); sys.exit(1)

    mons = glfw.get_monitors()
    MONITOR_INDEX = min(MONITOR_INDEX, len(mons)-1)
    mon = mons[MONITOR_INDEX]
    mode = glfw.get_video_mode(mon)
    width, height = mode.size.width, mode.size.height

    print(f"{glfw.REFRESH_RATE=}, {mode.refresh_rate=}")

    # フルスクリーン独占
    glfw.window_hint(glfw.RED_BITS, mode.bits.red)
    glfw.window_hint(glfw.GREEN_BITS, mode.bits.green)
    glfw.window_hint(glfw.BLUE_BITS, mode.bits.blue)
    glfw.window_hint(glfw.REFRESH_RATE, mode.refresh_rate)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
    win = glfw.create_window(width, height, "Flash Grid", monitor=mon, share=None)
    if not win:
        print("window create failed"); glfw.terminate(); sys.exit(1)

    glfw.make_context_current(win)
    # VSyncオン（1 = 垂直同期待ち。240Hzなら1フレーム≈4.17ms）
    glfw.swap_interval(1)

    # シェーダ
    prog = gl.glCreateProgram()
    vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(vs, VERT_SRC); gl.glCompileShader(vs)
    if not gl.glGetShaderiv(vs, gl.GL_COMPILE_STATUS):
        raise RuntimeError(gl.glGetShaderInfoLog(vs).decode())
    gl.glShaderSource(fs, FRAG_SRC); gl.glCompileShader(fs)
    if not gl.glGetShaderiv(fs, gl.GL_COMPILE_STATUS):
        raise RuntimeError(gl.glGetShaderInfoLog(fs).decode())
    gl.glAttachShader(prog, vs); gl.glAttachShader(prog, fs); gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
    gl.glUseProgram(prog)

    u_cell = gl.glGetUniformLocation(prog, "u_cell_px")
    u_line = gl.glGetUniformLocation(prog, "u_line_px")
    u_gamma = gl.glGetUniformLocation(prog, "u_gamma")
    u_view = gl.glGetUniformLocation(prog, "u_viewport")
    u_show = gl.glGetUniformLocation(prog, "u_show_grid")

    gl.glViewport(0, 0, width, height)
    gl.glDisable(gl.GL_BLEND)
    
    # Create VAO and VBO for fullscreen triangle
    vertices = np.array([
        -1.0, -1.0,
         3.0, -1.0,
        -1.0,  3.0
    ], dtype=np.float32)
    
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
    
    pos_attrib = gl.glGetAttribLocation(prog, "position")
    gl.glEnableVertexAttribArray(pos_attrib)
    gl.glVertexAttribPointer(pos_attrib, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

    # ループ：OFF→ON（ON_FRAMESだけ）→OFF…
    frame_in_cycle = 0
    cycles = 0
    while not glfw.window_should_close(win) and cycles < CYCLES:
        show = 1 if (frame_in_cycle < ON_FRAMES) else 0

        gl.glUniform2f(u_cell, float(CELL_PX[0]), float(CELL_PX[1]))
        gl.glUniform1f(u_line, LINE_PX)
        gl.glUniform1f(u_gamma, GAMMA)
        gl.glUniform2f(u_view, float(width), float(height))
        gl.glUniform1i(u_show, show)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        glfw.swap_buffers(win)
        glfw.poll_events()

        frame_in_cycle += 1
        if frame_in_cycle >= ON_FRAMES + OFF_FRAMES:
            frame_in_cycle = 0
            cycles += 1

    glfw.terminate()

if __name__ == "__main__":
    main()
