# RoPE_triton

- Triton으로 빌드한 RoPE와 Transformer Engine으로 빌드한 RoPE를 비교하는 레포지토리입니다.        
- 동일한 환경에서 테스트하기 위해 도커 컨테이너를 사용합니다.         

## 도커 이미지 빌드

- 도커 이미지를 만들기 위해 `build_docker.sh` 스크립트를 실행합니다.             
- 레포지토리에 포함된 Dockerfile을 사용해 이미지를 빌드합니다.                 
- Ninja의 MAX_JOBS를 각자의 환경에 맞춰 늘려주어 빌드 속도를 높일 수 있습니다.                

```shell
$ bash build_docker.sh
```

## 도커 컨테이너 생성

- 도커 컨테이너를 생성하기 위해 `run_container.sh` 스크립트를 실행합니다.               

```shell
$ bash run_container.sh
```


## 파이썬 프로그램 실행

- `compare_triton_rope_with_te_rope.py`는 Triton으로 빌드한 RoPE와 Transformer Engine으로 빌드한 RoPE를 비교하는 파이썬 프로그램입니다.     
- 두 방식의 실행 속도를 측정하며, 두 방식의 RoPE이 동일한 결과물을 생성하는지 비교하는 코드입니다.

```shell
$ python compare_triton_rope_with_te_rope.py
```

## 코드 설명

- Triton으로 작성된 RoPE 코드는 Unsloth의 코드를 참고하여 작성하였습니다.         
  - ref : https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rope_embedding.py      

- Triton으로 작성한 코드의 구성은 다음과 같습니다.        
1. RoPEKernel        
2. RoPEFunction
3.apply_rotary_pos_emb_triton


## 성능

- seq_len에 따라 각 RoPE의 실행시간을 측정한 결과는 다음과 같습니다.


```shell
rope-performance:
   seq_len    Triton     Torch
0     32.0  0.007388  0.006284
1     64.0  0.009204  0.008294
2    128.0  0.013212  0.012223
3    256.0  0.021615  0.020099
4    512.0  0.038205  0.035189
tokens_triton: tensor([-1.2466,  1.9525, -0.5653, -0.8070, -0.0448, -1.2624,  1.2703,  0.2500,
        -0.5281, -0.1046], device='cuda:0')
tokens_te: tensor([-1.2466,  1.9525, -0.5653, -0.8070, -0.0448, -1.2624,  1.2703,  0.2500,
        -0.5281, -0.1046], device='cuda:0')
Traceback (most recent call last):
  File "/app/compare_triton_rope_with_te_rope.py", line 260, in <module>
    main()
  File "/app/compare_triton_rope_with_te_rope.py", line 255, in main                                                                                                                                                                                                                                  assert torch.allclose(tokens_te, tokens_triton)                                                                                                                                                                                                                                               AssertionError  

```
