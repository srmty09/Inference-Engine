#include <iostream>
#include <cmath>
using namespace std;

void linear(int B, 
            int T, 
            int dim1, 
            int dim2,
            float* inp, 
            float* weights, 
            float* bias,
            float* out
){
    for(int b = 0; b < B; b++){
        for(int t = 0; t < T; t++){
            float* xid = inp+b*T*dim1+t*dim1;
            float* yid = out + b*T*dim2 + t*dim2;
            for(int v = 0;v < dim2;v++){
                float* wid = weights + v*dim1;
                float o = bias ? bias[v] : 0.0f;
                for (int c = 0; c < dim1; c++) {
                    o += xid[c] * wid[c];
                }

                yid[v] = o;
            }
        }
    }
}


// for feed forward block
void mlp(int B,
        int T, 
        int C, 
        int hidden_dim, 
        float* inp,
        float* out,
        float* c_fc, // this upscaling weights
        float* c_fc_bais, // this upscaling bias
        float* c_proj, // this is downscaling weights
        float* c_proj_bais, // this downscaling bias
        float* inter_preact, // output from the upscaling layer and without activation
        float* inter_act
){
    // upscaling from C to hidden_dim
    linear(B,T,C,hidden_dim,inp,c_fc,c_fc_bais,inter_preact);
    
    // gelu activation
    gelu(inter_preact,inter_act,B*T*hidden_dim);

    // todo dropout

    //downscaling from hidden_dim to C
    linear(B,T,hidden_dim,C,inter_act,c_proj,c_proj_bais,out);
};


void mha(
    float* c_attn,        // (3C, C)
    float* c_attn_bias,   // (3C)
    float* inp,           // (B, T, C)
    float* qkv,           // (B, T, 3C)
    float* preattn,       // (B, NH, T, T)
    float* attn,          // (B, NH, T, T)
    float* out,           // (B, T, C)
    int B,
    int T,
    int C,
    int NH,
    float* wo             // c_proj weight (C, C)
){
    int hs = C / NH;
    float scale = 1.0f / sqrtf(hs);

    linear(B, T, C, 3*C, inp, c_attn, c_attn_bias, qkv);
    for(int b = 0; b < B; b++){
        for(int h = 0; h < NH; h++){

            float* Q = qkv + b*T*3*C + h*hs;
            float* K = qkv + b*T*3*C + C + h*hs;
            float* scores = preattn + b*NH*T*T + h*T*T;
            linear(
                1, T, hs, T,
                Q, K, nullptr, scores
            );

            for(int i = 0; i < T*T; i++){
                scores[i] *= scale;
            }
            // causal mask 
            for(int t = 0; t<T;t++){
                for(int t2 = t+1; t2<T;t2++){
                    scores[t*T+t2] = -INFINITY; 
                }
            }

            // softmax
            softmax();

        }
    }    


    for(int b = 0; b < B; b++){
        for(int h = 0; h < NH; h++){

            float* A = attn + b*NH*T*T + h*T*T;
            float* V = qkv + b*T*3*C + 2*C + h*hs;
            float* O = out + b*T*C + h*hs;

            // A: (T, T)
            // V: (T, hs)
            linear(
                1, T, T, hs,
                A, V, nullptr, O
            );
        }
    }

    linear(B, T, C, C, out, wo, nullptr, out);
}


void softmax(){

}

#define layer_norm_eps 1e-5f

void layernorm(
    float* prenorm,
    float* postnorm,
    int B,
    int T,
    int C,
    float* ln_weights, // gamma
    float* ln_bias     // beta
){
    for(int b = 0; b < B; b++){
        for(int t = 0; t < T; t++){

            float* xid = prenorm  + b*T*C + t*C;
            float* yid = postnorm + b*T*C + t*C;

            // mean
            float mean = 0.0f;
            for(int c = 0; c < C; c++){
                mean += xid[c];
            }
            mean /= C;

            // variance
            float var = 0.0f;
            for(int c = 0; c < C; c++){
                float diff = xid[c] - mean;
                var += diff * diff;
            }
            var /= C;

            float inv_std = 1.0f / sqrtf(var + layer_norm_eps);

            // normalized
            for(int c = 0; c < C; c++){
                yid[c] = ln_weights[c] * (xid[c] - mean) * inv_std
                       + ln_bias[c];
            }
        }
    }
}


// for residual connection
void addNorm(
    float* residual,   // x
    float* prenorm,    
    float* out,        
    int B,
    int T,
    int C,
    float* postnorm,   
    float* ln_weights, // gamma
    float* ln_bias     // beta
){
    layernorm(prenorm, postnorm, B, T, C, ln_weights, ln_bias);

    int N = B * T * C;
    for(int i = 0; i < N; i++){
        out[i] = residual[i] + postnorm[i];
    }
}


// softmax activation function
void softmax(){

};

// gelu activation function
// GELU(x)=0.5∗x∗(1+Tanh( root(2/pi)∗(x+0.044715∗x 3)))
#define GeluScalingFactor sqrtf(2.0/M_PI)
void gelu(float* input, float* output,int N){
    for(int i = 0; i<N;i++){
        float x = input[i];
        float cube_x = x*x*x;
        float out = 0.5*x*(1+tanhf(GeluScalingFactor*(x+0.044715*cube_x)));
        output[i] = out;
    }
};

// for Linear block after the attention layers
void lm_head(
    int vocab_size,
    int B,
    int T,
    int C,
    float* weight,
    float* inp,
    float* out,
    float* bias
) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {

            float* xid = inp + b*T*C + t*C;
            float* yid = out + b*T*vocab_size + t*vocab_size;

            for (int v = 0; v < vocab_size; v++) {
                float* wid = weight + v*C;
                float o = bias ? bias[v] : 0.0f;

                for (int c = 0; c < C; c++) {
                    o += xid[c] * wid[c];
                }

                yid[v] = o;
            }
        }
    }
}



int main(){
    // float* inp = new float[1];
    // inp[0] = 1;
    // float* out = new float[1];
    // int N = 1;
    // gelu(inp,out,N);
    // for(int i = 0; i<N;i++){
    //     cout<<inp[i]<<" "<<out[i]<<"\n";
    // }


    // float* inp = new float[70];
    // float* out = new float[100];
    // float* weight = new float[70];
    // float* bias = nullptr;
    // int B = 2;
    // int T = 5;
    // int C = 7;
    // int V = 10;
    // lm_head(V,B,T,C,weight,inp,out,bias);

    return 0;
}






