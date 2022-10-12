# Enhanced_selectivity11
Python algorithm for solving Schrödinger equations for the model presented in PRA 97 063859 (2018) by Dr. Hisaki Oka.

In the aforementioned paper, there is a set of differential equations extracted from the usual time-dependent Schrödinger equations, given by:


$$ \frac{d}{dt}\psi ^{(2p)}(k, k', t) = -i(k+k')\psi ^{(2p)}(k, k', t)-i\sum _{\nu}\frac{1}{\sqrt{2}}\gamma _{\nu}[\psi ^{(1pm)}(k, \nu ,t) + \psi ^{(1pm)}(k', \nu ,t)] $$

$$ \frac{d}{dt}\psi ^{(1pm)}(k, \nu ,t) = -i(k+\omega _{m _{\nu}})\psi ^{(1pm)}(k, \nu ,t) - i \sqrt{2}\gamma _{\nu} \int dk'  \psi ^{(2p)}(k, k', t)- i \sum _{\nu '}\gamma _{\nu \nu '}\psi ^{(e)}(\nu ', t) $$

$$ \frac{d}{dt}\psi ^{(e)}(\nu ', t) = -i \omega _{e_{\nu '}}\psi ^{(e)}(\nu ', t)-i\sum _{\nu} \gamma _{\nu \nu '}\int dk \, \psi ^{(1pm)}(k, \nu ,t) $$

In order to solve the equations, we discretize the the photon fields. The discretized equations have the following form:

$$ \frac{d}{dt} \psi^{(2p)}_{kk'} (t)=-i(k+k')\psi ^{(2p)}_{kk'}(t)-i\sum_{\nu}\sqrt{\frac{\gamma F_{\nu}}{2\pi}}\left[ \psi^{(1pm)}_{k\nu}(t) + \psi^{(1pm)}_{k'\nu}(t) \right] $$

$$ \frac{d}{dt} \psi ^{(1pm)}_{k\nu}(t) = -i(k+\omega _{m_{\nu}})\psi ^{(1pm)}_{k\nu}(t) -i \sqrt{\frac{2\gamma F_{\nu}}{\pi}}\sum _{k'}\psi ^{(2p)}_{kk'}(t) -i\sum _{\nu '}\sqrt{\frac{\gamma F_{\nu \nu '}}{\pi}}\psi ^{(e)}_{\nu '}(t) $$

$$ \frac{d}{dt}\psi ^{(e)}_{\nu '}(t) = -i\omega _{e_{\nu}} \psi ^{(e)}_{\nu '}(t)-i \sum _{\nu \nu'}\sqrt{\frac{\gamma F_{\nu \nu '}}{\pi}} \sum _{k} \psi ^{(1pm)}_{k\nu}(t) $$

Defining the following matrices which are containing the functions for each state

 $$ \psi ^{(2p)} = 
\begin{pmatrix}
\psi ^{(2p)} _{k_{0}k_{0}} & \psi ^{(2p)} _{k_{0}k_{1}} & \cdots & \psi ^{(2p)} _{k_{0}k_{M}} \\
\psi ^{(2p)} _{k_{1}k_{0}} & \psi ^{(2p)} _{k_{1}k_{1}} & \cdots & \psi ^{(2p)} _{k_{1}k_{M}} \\
\vdots & \vdots & \cdots & \vdots \\
\psi ^{(2p)} _{k_{M}k_{0}} & \psi ^{(2p)} _{k_{M}k_{1}} & \cdots & \psi ^{(2p)} _{k_{M}k_{M}} \\
\end{pmatrix} $$

$$ \psi ^{(1pm)} = 
\begin{pmatrix}
\psi ^{(1pm)} _{k_{0},0} & \psi ^{(1pm)} _{k_{0},1} & \cdots & \psi ^{(1pm)} _{k_{0},N} \\
\psi ^{(1pm)} _{k_{1},0} & \psi ^{(1pm)} _{k_{1},1} & \cdots & \psi ^{(1pm)} _{k_{1},N} \\
\vdots & \vdots & \cdots & \vdots \\
\psi ^{(1pm)} _{k_{M},0} & \psi ^{(1pm)} _{k_{M},1} & \cdots & \psi ^{(1pm)} _{k_{M}, N} \\
\end{pmatrix} $$

$$ \psi ^{e} = 
\begin{pmatrix}
\psi ^{e} _{0} \\
\psi ^{e} _{1} \\
\vdots \\
\psi ^{e} _{N} \\
\end{pmatrix} $$

$$ \gamma = 
\begin{pmatrix}
\gamma _{0} \\
\gamma _{1} \\
\vdots \\
\gamma _{N} \\
\end{pmatrix} $$

$$ \gamma ^{(gm)} = 
\begin{pmatrix}
\gamma _{0} & \gamma _{1} & \cdots & \gamma _{N} \\
\gamma _{0} & \gamma _{1} & \cdots & \gamma _{N} \\
\vdots & \vdots & \cdots & \vdots \\
\gamma _{0} & \gamma _{1} & \cdots & \gamma _{N} \\
\end{pmatrix} $$

$$ \gamma ^{(me)} = 
\begin{pmatrix}
\gamma _{00} & \gamma _{01} & \cdots & \gamma _{0N} \\
\gamma _{10} & \gamma _{11} & \cdots & \gamma _{1N} \\
\vdots & \vdots & \cdots & \vdots \\
\gamma _{N0} & \gamma _{N1} & \cdots & \gamma _{NN} \\
\end{pmatrix} $$

the discretized form of the set of differential equations is given by

$$ \frac{d}{dt}\psi ^{(2p)}_{k_{\alpha}k_{\beta}}=-i(k_{\alpha}+k_{\beta})\psi ^{(2p)}_{k_{\alpha}k_{\beta}} - i (\psi ^{(1pm)}\gamma)_{\alpha} - i (\psi ^{(1pm)}\gamma)_{\beta} $$

$$  \frac{d}{dt}\psi ^{(1pm)}_{k_{\alpha},j}=-i(k_{\alpha}+\omega_{m_{j}}) \psi ^{(1pm)}_{k_{\alpha},j} - i (\psi ^{(2p)}\gamma ^{(gm)})_{\alpha j} - i(\gamma ^{(me)}\psi ^{e})_{j}  $$

$$ \frac{d}{dt}\psi ^{e}_{j} = -i \omega _{e _{j}}\psi ^{e}_{j}- i \sum _{n=0}^{N}(\psi ^{(1pm)}\gamma ^{(me)})_{nj} $$

This configuration of arrays is contained in the code bellow.
