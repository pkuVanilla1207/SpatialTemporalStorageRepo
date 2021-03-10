import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mc

def check_and_make_dir(dir):
    if os.path.exists(dir)==False:
        os.makedirs(dir)

def read_text(path):
    with open(path,'r') as f:
        text=f.read()
        text=text.replace('\n', '')
        return text
    
def get_split_text(text):
    pattern = re.compile(r'[\-|\d|E|.]+')
    result = pattern.findall(text)
    return result

def text_to_matrix_1d(text):
    result = get_split_text(text)
    matrix=np.array(result).astype(np.float)
    return matrix

def text_to_matrix_2d(text,shape):
    matrix_1d=text_to_matrix_1d(text)
    matrix_2d=matrix_1d.reshape(shape)
    return matrix_2d

def text_to_matrix_3d(text,shape):
    matrix_1d=text_to_matrix_1d(text)
    count=shape[0]*shape[1]
    depth_matrixs=[matrix_1d[i*count:i*count+count] for i in range(shape[2])]
    matrix_3d=np.stack(depth_matrixs,axis=1)
    matrix_3d=matrix_3d.reshape(shape)
    return matrix_3d

def get_file_matrix(path):
    name=os.path.splitext(os.path.basename(path))[0]
    if name in 'hstuv' or name == 'out1':
        text=read_text(path)
        if name in 'hstuv':
            if name=='h':
                matrix=text_to_matrix_2d(text,(638,524))[::-1,::-1]
            else:
                matrix=text_to_matrix_3d(text,(638,524,33))[::-1,::-1,:]
        else:
            text=re.search(r'(?<=depth).*',text)[0]
            matrix=text_to_matrix_1d(text)
        return matrix
    else:
        return None
    
def get_lons_lats_depths(path):
    matrix=get_file_matrix(path)
    lons=matrix[0:524]
    lats=matrix[524:524+638]
    depths=matrix[524+638:524+638+33]
    return lons,lats,depths

def get_linear_colormap(values,colors):
    colors=np.array(colors)/255
    norm=plt.Normalize(min(values),max(values))
    tuples = list(zip(map(norm,values), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    return cmap

def get_colormap(values,colors):
    colors=np.array(colors)/255
    cmap = mc.ListedColormap(colors)
    norm=mc.BoundaryNorm(values, cmap.N)
    return cmap,norm

def show_colormap(ax,plt,**kwargs):
    i=kwargs['index']
    matrix=kwargs['matrix']
    values=kwargs['values']
    colors=kwargs['colors']
    bg_value=kwargs['bg_value']
    bg_color=kwargs['bg_color']
    border=kwargs['border']
    matrix=matrix[:,:,i] if i!=-1 else matrix
    matrix=matrix.copy()
    cmap,norm=get_colormap(values,colors)
    matrix[matrix==bg_value]=np.nan
    cmap.set_bad(color=np.array(bg_color)/255)
    values=np.array(values)
    im=ax.imshow(matrix,cmap=cmap,norm=norm,vmax=np.max(values),vmin=np.min(values))
    values=values if border==True else values[1:-1]
    cbar = plt.colorbar(im,ticks=values)
    return ax,plt

def show_stream(ax,plt,**kwargs):
    u=kwargs['u']
    v=kwargs['v']
    i=kwargs['index']
    interval=kwargs['interval']
    bg_value=kwargs['bg_value']
    bg_color=kwargs['bg_color']
    bg_color=bg_color+[255]
    u=u[:,:,i] if i!=-1 else u
    v=v[:,:,i] if i!=-1 else v
    bg=np.zeros((u.shape[0],u.shape[1],4),dtype=np.uint8)
    bg[u==bg_value]=bg_color
    ax.imshow(bg)
    x=np.arange(u.shape[1])
    y=np.arange(u.shape[0])
    x=x[::interval]
    y=y[::interval]
    u=u[::interval,::interval]
    v=v[::interval,::interval]
    plt.streamplot(x,y,u,v)
    return ax,plt

def output_fig(shape,output_path,scale,dpi,func,**kwargs):
    plt_size = ((shape[1]/dpi)*scale,(shape[0]/dpi)*scale)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=plt_size, dpi=dpi)
    if func==show_colormap:
        ax,plt=show_colormap(ax,plt,**kwargs)
    else:
        ax,plt=show_stream(ax,plt,**kwargs)
    plt.savefig(output_path)
    plt.close('all')
    
def output_all_img(shape,output_dir,output_name,scale,dpi,func,**kwargs):
    if len(shape)==3:
        for i in range(shape[2]):
            kwargs['index']=i
            output_path=os.path.join(output_dir,output_name+'_'+str(i)+'.jpg')
            output_fig(shape,output_path,scale,dpi,func,**kwargs)
    else:
        kwargs['index']=-1
        output_path=os.path.join(output_dir,output_name+'.jpg')
        output_fig(shape,output_path,scale,dpi,func,**kwargs)
        
def output_all_visual(matrix,output_dir,output_name,values,colors,scale,dpi):
    output_all_img(matrix.shape,output_dir,output_name,scale,dpi,show_colormap,
                   matrix=matrix,values=values,colors=colors,bg_value=9999,
                   bg_color=[50,50,50],border=False)

def get_color_values(vmin,vmax,length,vborder=None):
    length-=1
    values=np.arange(vmin,vmax+(vmax-vmin)/length,(vmax-vmin)/length)
    if vborder!=None:
        values=[-vborder]+values.tolist()+[vborder]
    return values

def output_visual_other_img(matrix,output_dir,output_name,scale,dpi):
    vmin=np.min(matrix)
    vmax=np.max(matrix[matrix!=9999])
    values=get_color_values(vmin,vmax,12,9999)
    colors=[[161,0,199],[110,0,219],[31,61,250],[0,161,230],
            [0,199,199],[0,209,140],[0,219,0],[161,230,51],
            [230,219,51],[230,176,46],[240,130,41],[240,0,0],[219,0,99]]
    output_all_visual(matrix,output_dir,output_name,values,colors,scale,dpi)

def output_visual_temperature_img(matrix,output_dir,output_name,scale,dpi):
    values=[-100,17,18,19,20,21,22,24,25,26,27,28,29,100]
    colors=[[161,0,199],[110,0,219],[31,61,250],[0,161,230],
            [0,199,199],[0,209,140],[0,219,0],[161,230,51],
            [230,219,51],[230,176,46],[240,130,41],[240,0,0],[219,0,99]]
    output_all_visual(matrix,output_dir,output_name,values,colors,scale,dpi)
    
def output_steam_img(u,v,output_dir,output_name,scale,dpi):
    output_all_img(u.shape,output_dir,output_name,scale,dpi,show_stream,
                   u=u,v=v,bg_value=9999,bg_color=[50,50,50],interval=30)
    
def start_output(file_paths,output_dir,scale=2,dpi=100):
    makedir=False
    if len(file_paths)==2:
        u=get_file_matrix(file_paths[0])
        v=get_file_matrix(file_paths[1])
        filename='sr'
        makedir=True
    else:
        matrix=get_file_matrix(file_paths[0])
        filename=os.path.splitext(os.path.basename(file_paths[0]))[0]
        if len(matrix.shape)==3:
            makedir=True
    if makedir==True:
        output_dir=os.path.join(output_dir,filename)
        check_and_make_dir(output_dir)
    if filename=='t':
        output_visual_temperature_img(matrix,output_dir,filename,scale,dpi)
    elif filename=='sr':
        output_steam_img(u,v,output_dir,filename,scale,dpi)
    else:
        output_visual_other_img(matrix,output_dir,filename,scale,dpi)
    

if __name__=='__main__':
    #输出风场文件路径
    file_paths=[r'C:\Users\Alan\Desktop\ocean\u.txt',
                r'C:\Users\Alan\Desktop\ocean\v.txt']
    #输出其他文件路径
    #file_paths=[r'C:\Users\Alan\Desktop\ocean\t.txt']
    #输出文件夹
    output_dir=r'C:\Users\Alan\Desktop'
    #开始输出
    start_output(file_paths,output_dir)