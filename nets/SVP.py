import tensorflow as tf
from tensorflow.python.framework import function

def removenan(x):
    return tf.where(tf.is_finite(x), x,tf.zeros_like(x))




@tf.RegisterGradient('Svd_')
def gradient_svd(op, ds, dU, dV):
    s, U, V = op.outputs

    u_sz = tf.squeeze(tf.slice(tf.shape(dU),[1],[1]))
    v_sz = tf.squeeze(tf.slice(tf.shape(dV),[1],[1]))
    s_sz = tf.squeeze(tf.slice(tf.shape(ds),[1],[1]))

    S = tf.matrix_diag(s)
    s_2 = tf.square(s)

    eye = tf.expand_dims(tf.eye(s_sz),0) 
    k = (1 - eye)/(tf.expand_dims(s_2,2)-tf.expand_dims(s_2,1) + eye)
    KT = tf.matrix_transpose(k)
    KT = removenan(KT)
    
    def msym(X):
        return (X+tf.matrix_transpose(X))
    
    def left_grad(U,S,V,dU,dV):
        U, V = (V, U); dU, dV = (dV, dU)
        D = tf.matmul(dU,tf.matrix_diag(1/(s+1e-8)))
    
        grad = tf.matmul(D + tf.matmul(U, tf.matrix_diag(tf.matrix_diag_part(-tf.matmul(U,D,transpose_a=True)))
                           + 2*tf.matmul(S, msym(KT*(-tf.matmul(D,tf.matmul(U,S),transpose_a=True))))) ,V,transpose_b=True)
        
        grad = tf.matrix_transpose(grad)
        return grad

    def right_grad(U,S,V,dU,dV):
        grad = tf.matmul(2*tf.matmul(U, tf.matmul(S, msym(KT*(tf.matmul(V,dV,transpose_a=True)))) ),V,transpose_b=True)
        return grad
    
    grad = tf.cond(tf.greater(v_sz, u_sz), lambda : left_grad(U,S,V,dU,dV), 
                                           lambda : right_grad(U,S,V,dU,dV))
    
    return [grad]

def gradient_eid(op, ds, dU, dV):
    return gradient_svd(op, ds, dU, dV) + [None]*3

@function.Defun(tf.float32, tf.float32,tf.float32,tf.float32,func_name = 'EID', python_grad_func = gradient_eid)
def SVD_grad_map(x, s, u, v):
    return s,u,v 

