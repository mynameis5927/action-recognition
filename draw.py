import numpy as np
import matplotlib.pyplot as plt

f = open('input.txt','r')
lines = f.readlines()
f.close()

#print(lines)

is_loss = True


Y = []
x = []
for l in lines:
    l = l.replace('\\','')
    #print(l)
    
    itm = l.split('&')
    print(itm)
    
    if is_loss:
        Y.append([float(itm[2]),float(itm[4]),float(itm[6])])
    else:
        Y.append([float(itm[1]),float(itm[3]),float(itm[5])])
    
    x.append(float(itm[0]))
    
    

Y = np.array(Y)
x = np.array(x)

print(Y)

# ========== 绘图设置 ==========
plt.figure(figsize=(10, 6))  # 设置画布大小

# 绘制三条曲线
plt.plot(x, Y[:,0], label='LSTM', color='blue', linestyle='-')
plt.plot(x, Y[:,1], label='ST-GAN', color='red', linestyle='--')
plt.plot(x, Y[:,2], label='ST-GAN-Attention', color='green', linestyle=':')

# ========== 样式设置 ==========
#plt.title('Comparison on accuracy of three algorithms', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
if is_loss:
    plt.ylabel('Loss', fontsize=12)
else:
    plt.ylabel('Accuracy', fontsize=12)
plt.legend()
# 显示图形
plt.tight_layout()  # 自动调整布局
#plt.show()

if is_loss:
    plt.savefig('comparsions-loss.png')
else:
    plt.savefig('comparsions-accuracy.png')





    
    
    