把“所有可能的无人机轨迹”参数化

FY1 初始位置是：

$$ P_0=(17800,0,1800) $$

题目规定：

等高度
匀速
直线飞行
速度范围 \(70\sim140\) m/s

所以实际上无人机轨迹并不是任意的。

它只需要两个东西：

① 飞行方向

假设水平面内方向角为 \(\theta\)。

那么单位方向向量可以写成：

$$ \boldsymbol e=(\cos\theta,\sin\theta,0) $$
② 飞行速度
$$ 70\leq v\leq140 $$

于是无人机轨迹就完全确定：

$$ \boxed{ U(t)=P_0+vt\boldsymbol e } $$

也就是：

$$ U(t)= (17800+vt\cos\theta,\, vt\sin\theta,\, 1800) $$
四、注意：这时候“轨迹不确定”其实已经变成了“参数不确定”

这是我们建模时非常重要的一步。

不要再说：

无人机轨迹是未知的。

而应该说：

无人机轨迹由飞行方向 \(\theta\) 和速度 \(v\) 参数化。

于是：

$$ \boxed{ \text{轨迹} \longleftrightarrow (\theta,v) } $$

这一下问题就从一个“无限复杂的轨迹规划问题”，变成了一个有限维参数优化问题。

五、然后再看投弹

假设：

$$ t_d=\text{投弹时间} $$

那么投弹位置就是：

$$ P_d=U(t_d) $$

所以：

$$ P_d= (17800+vt_d\cos\theta,\, vt_d\sin\theta,\, 1800) $$

你会发现一个非常漂亮的关系：

$$ \boxed{ (\theta,v,t_d) \rightarrow \text{投弹位置} } $$
六、然后再看起爆

假设投弹以后经过：

$$ \tau $$

秒起爆。

那么：

$$ t_e=t_d+\tau $$

烟幕弹从投弹点开始做：

水平速度继承无人机速度；
竖直方向受到重力。

因此起爆点：

$$ P_e= P_d+ v\tau\boldsymbol e + \begin{pmatrix} 0\\ 0\\ -\frac12g\tau^2 \end{pmatrix} $$

把 \(P_d\) 展开：

$$ \boxed{ P_e= P_0+ v(t_d+\tau) (\cos\theta,\sin\theta,0) + (0,0,-\frac12g\tau^2) } $$

于是：

$$ \boxed{ (\theta,v,t_d,\tau) \rightarrow \text{起爆位置} } $$
七、这时候你就会发现：第二问真正的“优化变量”已经出现了

可以定义：

$$ \boxed{ \boldsymbol{x} = (\theta,v,t_d,\tau) } $$

也就是说：

变量	含义
\(\theta\)	无人机飞行方向
\(v\)	无人机飞行速度
\(t_d\)	投弹时间
\(\tau\)	投弹后起爆延迟

然后整个系统就变成：

$$ \boldsymbol{x} $$

↓

无人机轨迹

↓

投弹位置

↓

烟幕弹轨迹

↓

起爆位置

↓

烟幕云团轨迹

↓

导弹—目标视线

↓

遮蔽时间

所以最后：

$$ \boxed{ T_{\rm shield} = F(\theta,v,t_d,\tau) } $$

第二问实际上就是：

$$ \boxed{ \max_{\theta,v,t_d,\tau} F(\theta,v,t_d,\tau) } $$
