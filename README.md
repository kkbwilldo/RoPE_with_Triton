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

### 1. RoPEKernel        
  - RoPE의 forward와 backward 연산을 수행하는 Triton 커널입니다.        
  - 토큰의 포인터,  freq의 포인터, 오프셋을 계산하기 위한 값 그리고 블록 크기를 입력받아 다음의 연산을 수행합니다.        
  
  ```python
  token_embed = (token * cos) + (rotate_half(token) * sin)
  ```

  - 그리고 해당 연산의 backward도 구현하였습니다.           

### 2. RoPEFunction              
  - RoPE의 forward와 backward 연산을 수행하는 클래스입니다.               
  - autograd.Function을 상속받아 새로운 autograd 함수를 만들었습니다.          
  - tokens와 freqs 그리고 ctx(컨텍스트 객체)를 입력받은 후 RoPEKernel의 forward 및 backward를 호출하여 RoPE 연산을 수행합니다.      
  - ctx에 backward에서 사용되는 정보를 저장하여 backward에서 사용합니다.           

### 3.apply_rotary_pos_emb_triton               
  - RoPE 연산의 forward를 호출하는 함수입니다.      
  - 2 가지 tensor_format("bshd", "sbhd")를 지원합니다.     
  - 임베딩 된 텐서를 반환합니다.        

### 4. benchmark                  
  - RoPE의 연산 성능을 측정하는 함수입니다.       
  - 성능은 연산의 실행 시간을 말하며, sequence length가 커짐에 따라 성능이 어떻게 변하는지를 확인합니다.       
  - 여기서 Triton으로 작성한 RoPE의 성능과 Transformer Engine으로 작성한 RoPE의 성능을 비교합니다.            

### 5. allclose
  - Triton으로 작성한 RoPE와 Transformer Engine으로 작성한 RoPE가 동일한 입력에 대해 동일한 출력을 생성하는지 확인합니다.      
  - 현재 torch.allclose에서 assertion이 발생하고 있으며, 아직 원인 파악 중입니다.          

## 성능

- seq_len에 따라 각 RoPE의 실행시간을 측정한 결과는 다음과 같습니다.


```shell
rope-performance:
   seq_len    Triton  Transformer Engine
0     32.0  0.007408            0.006168
1     64.0  0.008943            0.008085
2    128.0  0.013009            0.012058
3    256.0  0.021409            0.020198
4    512.0  0.038282            0.035214
```

- 위 수치의 단위는 ms입니다.         
- 미세하게 Triton의 성능이 더 느린 것을 확인했습니다.            
- 대략 0.01ms정도 차이나며 이는 sequence length가 더 커짐에 따라 더 차이날 수 있을 것으로 생각됩니다.

```shell
tokens_triton: tensor([ 0.6490,  2.5151,  0.0706,  1.1732,  1.0390, -0.1717,  0.0655, -0.7748,
        -0.8035, -1.6004], device='cuda:0')
tokens_te: tensor([ 0.6490,  2.5151,  0.0706,  1.1732,  1.0390, -0.1717,  0.0655, -0.7748,
        -0.8035, -1.6004], device='cuda:0')
Traceback (most recent call last):
  File "/app/compare_triton_rope_with_te_rope.py", line 260, in <module>
    main()
  File "/app/compare_triton_rope_with_te_rope.py", line 255, in main
    assert torch.allclose(tokens_te, tokens_triton)
AssertionError
```

- 현재 두 구현이 동일한 입력에 대해서 다른 출력을 생성하고 있습니다.            
- 원인에 대해서는 아직 파악 중입니다.            
