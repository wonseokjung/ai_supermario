# A.I_Supermario
강화학습을 이용한 슈퍼마리오 만들기 튜토리얼 입니다. 


아래의 설명을 읽으시고 환경을 설치하신 뒤, 
제 깃헙을 클론하셔서 아래의 파이썬 코드를 실행하시면 됩니다. 


1_sb_ppo_agent.py -> PPO를 사용하여 학습 


8_sb_dqn_supermari.py	- > DQN을 사용하여 학습 


## 또한 google colab을 사용하여,

DQN으로 슈퍼마리오, Pong 게임을 학습시킬수 있는 Jupyter notebook도 함께 넣어놨습니다. 

아래의 링크를 통하여 다른 설치 없이 바로 Colab상에서 GPU를 사용하여 학습시키는 것이 가능합니다. 


<img src="https://www.dropbox.com/s/n7ypkshytfrc53y/Screenshot%202018-11-02%2009.29.36.png?raw=1">


https://colab.research.google.com/github/wonseokjung/ai_supermario/blob/master/4_pong_dqn%20(1).ipynb


<img src="https://www.dropbox.com/s/uhonvggdut4kmt2/Screenshot%202018-11-02%2009.30.46.png?raw=1">

https://colab.research.google.com/github/wonseokjung/ai_supermario/blob/master/2_supermario_dqn.ipynb




### Prerequisites

Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows

#### Ubuntu

```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X

Installation of system packages on Mac requires [Homebrew](https://brew.sh/). With Homebrew installed, run the follwing:

```
brew install cmake openmpi
```

### Install using pip

Install the Stable Baselines package

Using pip from pypi:

```
pip install stable-baselines
```

[![Mario](https://user-images.githubusercontent.com/2184469/40949613-7542733a-6834-11e8-895b-ce1cc3af9dbb.gif)](https://user-images.githubusercontent.com/2184469/40949613-7542733a-6834-11e8-895b-ce1cc3af9dbb.gif)



# Installation

The preferred installation of `gym-super-mario-bros` is from `pip`:

```
pip install gym-super-mario-bros
```



<img src="https://www.dropbox.com/s/9bxgpvp3ynvlelk/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.002.jpeg?raw=1">

더 자세히 강화학습으로 슈퍼마리오 에이전트를 만드는  튜토리얼을

시리즈로 만들어 보면 더 많은 분께 도움이 될것이라 생각될 생각하여 시작하게 되었습니다. 


Video lecture : 

https://www.youtube.com/watch?v=ydCrd9cDLsU



github : 

https://github.com/wonseokjung/A.I_Supermario

slide : 

https://github.com/wonseokjung/A.I_Supermario/blob/master/Tutorial/1_Environment_Install/%EC%8A%88%ED%8D%BC%EB%A7%88%EB%A6%AC%EC%98%A4%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BCpdf.pdf




첫번째로  슈퍼마리오 환경과 학습 방법에 대해서 설명하도록 하겠습니다. 



<img src="https://www.dropbox.com/s/ro8hii3qqu6euzi/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.005.jpeg?raw=1">



<img src="https://www.dropbox.com/s/a51z92gw9zo6224/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.006.jpeg?raw=1">









<img src="https://www.dropbox.com/s/tpy22td4fc6shey/Screenshot%202018-08-30%2021.31.16.png?raw=1"> 

Reinforcement Learning은 Markov Decision Process를 통하여 의사결정을 합니다. 



State $S_t$에서 Agent는 Action $A_t$를 선택하며, 

그 Action $A_t$를 Environment이 받고 

그 Action $A_t$ 에 대한 Reward $R$과 다음 state $S_{t+1}$를 return합니다. 

Agent는 다시 그 state $S_{t+1}$ 에서 Action $A_{t+1}$를 선택하여 환경에게 전달합니다. 

이런 과정을 여러번 거치면서 Agent는 특정 state에서 특정 action의 value ( 가치 )를 알게되며,

이와같은 학습을 거치며 이 리워드를 더 높게 받을 Action을 선택합니다. 



<img src="https://www.dropbox.com/s/v016t41ote1r1ww/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.007.jpeg?raw=1">



SuperMario 환경에서도 마찬가지로  Markov Decision Process를 통해 학습을 합니다. 

슈퍼마리오 게임 화면이 State 이며, Agent인 마리오는 화면을 인풋으로 받은 뒤 Action을 선택합니다. 



이 Action은  위,왼쪽,오른쪽,아래, A,B버튼으로 구성되어있습니다. 

<img src="https://www.dropbox.com/s/0hus0nsgtkskx3r/Screenshot%202018-08-30%2021.44.58.png?raw=1">



환경은 마리오가 선택한 Action을 받고 다음 state, 리워드 또는  페널티를 return 합니다. 





<img src="https://www.dropbox.com/s/tcp7ph6crur1ps0/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.008.jpeg?raw=1">



그렇다면 Action을 받고 reward,penalty 그리고 다음 State를 return해주는 환경은 무엇일까요? 



개인이 강화학습을 위한 슈퍼마리오 환경을 만들었고, 닌텐도 에뮬레이터를 통하여 이를 실행할수 있습니다. 

https://github.com/Kautenja/gym-super-mario-bros



<img src="https://www.dropbox.com/s/evc0zoemfvmban0/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.009.jpeg?raw=1">

<iframe width="560" height="315" src="https://www.youtube.com/embed/N8yyxEuiqDU" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

이 환경은 pip install gym-super-mario-bros 명령어를 통하여 받을수 있습니다.  

이 환경은 python 3.6 이상 버전에서만 설치가 되니 설치되어 있는 파이썬 버전을 확인해주세요. 



설치를 완료하신 뒤에 python3.6 을 열고 import gym_super_mario_bros 명령어로 슈퍼마리오 환경을 임포트 해줍니다. 



그 뒤 gym_super_mario_bros.make( ) 함수를 이용하여 환경을 만든뒤 env 변수로 선언을 해줍니다. 



여기서 make 함수 안에 들어가는 SuperMarioBros-v0는 슈퍼마리오 환경 버전입니다. 
바로 다음장에서 자세히 설명하도록 하겠습니다. 



env 변수로 선언을 해주셨으면 reset() 함수로 환경을 초기화 시키신 뒤 render() 함수를 불러주시면 
슈퍼마리오 환경이 생성됩니다. 

<img src="https://www.dropbox.com/s/54a5svhtxbnxebe/Screenshot%202018-08-30%2022.29.37.png?raw=1">

여기서 슈퍼마리오는 전혀움직이지 않는데 그 이유는 action을 선택해주는 코드를 넣지 않아 

환경이 다음 state를 return하지 않기 때문입니다. 



그럼 슈퍼마리오 환경에 대해서 조금 더 알아볼게요. 

<img src="https://www.dropbox.com/s/95pwyhhuiqfb5qr/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.010.jpeg?raw=1">



<img src="https://www.dropbox.com/s/ggri4zvp1so666k/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.011.jpeg?raw=1">

슈퍼마리오 환경은 총 8개의 World로 나누어져 있으며 각 World당 4개의 레벨이 존재합니다. 

각 월드, 레벨에는 특징이 있어서 해결해야하는 난이도가 다르며, 몬스터의 종류, 넘어야하는 장애물이 다릅니다. 



<img src="https://www.dropbox.com/s/y6k3wqb8vfxer0c/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.012.jpeg?raw=1">

gym_super_mario_bros.make( ) 함수를 사용하여 SuperMarioBros-v0 를 불러봤는데요. 



이 SuperMarioBros-v0 환경은  일반 게임플레이와 마찬가지로 월드1 - 레벨1에서 시작해서 

클리어했을때 다음 레벨로 넘어가는 환경입니다. 

하지만 특정 월드, 레벨을 make하고 싶다면 SuperMarioBros-world-level-v<verision> 의 형태에서 

world를 1~8 까지중 원하는 월드로 지정하고,

<img src="https://www.dropbox.com/s/wptgd8tkmjf32xg/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.013.jpeg?raw=1">

레벨 또한 1~4까지 숫자로 지정하여 원하는 월드와 레벨을 불러올수 있습니다. 



<img src="https://www.dropbox.com/s/6a9jk6yvisjqkg5/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.014.jpeg?raw=1">



맨 마지막의 version의 의미는 위의 그림과 같은데요. 

version1은 오리지날 그림이고, version2 는 background을 검은색으로 바꿨습니다. 

<img src="https://www.dropbox.com/s/iu7kz5mv1lc7e2v/Screenshot%202018-08-30%2022.51.31.png?raw=1">



월드1에서 레벨1, 레벨2에서 가장 큰 차이점은 "배경의 색이 다르다." 정도인데요. 

사람이 봤을때 이 배경의 색은 게임을 하는데 크게 다를점은 없지만, Input으로 게임 화면을 받아 학습을 하는 강화학습은

배경 색이 다름으로 인해 전혀 다른 상황이라고 판단하기 때문에 이를 방지하기 위해 배경의 색을 일치하게 만들어주는 것입니다. 



Version3은 배경의 퀄리티가 더 떨어지고 version4는 퀄리티도 더 낮으며 점수판도 제거하였습니다. 

이를 통해 마리오 agent가 화면, 장애물이  어떻게 생겼는지 자세히 알지 않아도 형태만 보고 행동을 잘 알수 있는지 실험해 볼수 있겠네요. 

<img src="https://www.dropbox.com/s/0bi0puo45z7ejm1/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.015.jpeg?raw=1">



이렇게 불러온 환경으로 Supermario는 환경의 화면을 보고 각 레벨 끝에 있는 깃발을 잡기위해 action 을 선택합니다. 

슈퍼마리오 에이전트가 깃발을 잡게 만들기 위해서 적절한 reward를 주는것이 중요한데요. 

<img src="https://www.dropbox.com/s/eh3sb0m49jfjezc/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.016.jpeg?raw=1">



저는 Reward 설정을 깃발에 가까워지면 reward를 주었습니다. 그리고 목표에 도착했을때 더 높은 값의 reward를 주었고요. 



반대로 목표에 달성하지 못하고, 시간이 지날때마다 penalty를 주었습니다. 또한 깃발에서 멀어지는 action을 선택했을때도 penalty를 주었습니다. 





<img src="https://www.dropbox.com/s/uquygakgvwgy9ia/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.017.jpeg?raw=1">



그렇다면 슈퍼마리오 에이전트를 DQN 알고리즘을 통하여 어떻게 학습을 하는지 전체적을 알아보도록 하겠습니다. 





<img src="https://www.dropbox.com/s/zm1sqnjl0bbe2ba/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.018.jpeg?raw=1">



 먼저 아까 env변수로 선언한 슈퍼마리오 환경에서 obsevation_space.shape 함수는 현재 supemairo 화면의 size를 return합니다. 

240,256,3  을 리턴하는데, 이 의미는 화면의 세로, 가로 채널의 크기 입니다. 



<img src="https://www.dropbox.com/s/o64ji55qtc7bybm/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.020.jpeg?raw=1">



그리고 action_space.n 함수는 현재 슈퍼마리오가 선택할수 있는 action의 조합입니다. 

 <img src="https://www.dropbox.com/s/0hus0nsgtkskx3r/Screenshot%202018-08-30%2021.44.58.png?raw=1">

이 action은 닌텐도 조이스틱의 버튼 조합인데요. 

위, 왼쪽,오른쪽,아래 버튼과 A,B 버튼 즉 점프, 불꽃발사의  조합입니다. 



<img src="https://www.dropbox.com/s/xbpihawm2hm9f1t/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.021.jpeg?raw=1">

하지만 실제로 우리가 필요한 버튼은 256가지나 되지는 않습니다. 

우리는 슈퍼마리오가 깃발을 향해 장애물과 괴물을 피하며 잘 달리는 행동을 학습하길 원하는 것이기 때문이죠. 

이렇게 action의 수가 많으면 학습하는데 시간이 오래 걸리게 됩니다. 

그래서 필요한 action의 조합으로 리스트를 만들어 이것을 

env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)로 환경에 적용해줍니다. 

<img src="https://www.dropbox.com/s/97zhrj5qbmb5r7x/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.022.jpeg?raw=1">



강화학습에서 현재 state에서 가장 action value가 높은 action을 선택하는, 즉 현재 상황에서 에이전트가 가장 좋다고 판단하는 행동을 하는 것을 Exploitation이라고 하며, 



반대로 현재 가장 좋은 action은 아니지만 탐험을 해보는 exploration은 강화학습으로 학습시킬때 매우 중요합니다.

Exploration을 하지 않는 에이전트는 점프도 하지 않고 계속 앞으로 가는 action만 선택하게 됩니다. 그 상황에서 당장 가장 좋은 action 은 앞으로 가는 action인 오른쪽 버튼을 누른는 것이기 때문입니다. 

<img src="https://www.dropbox.com/s/m7czd57zc8pc5pt/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.023.jpeg?raw=1">



그래서 epsilon값을 정하고 numpy의 np.random.rand() 함수를 통하여 난수발생을 합니다. 함수를 통해 리턴된 값이 epsilon보다 작으면 agent는 할수 있는  action들중 임의로 하나를 선택합니다. 
이렇게 하는 것은 exploration이라고 합니다. 



<img src="https://www.dropbox.com/s/49bim3msod6unkw/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.024.jpeg?raw=1">

반면에 epsilon값보다 높으면 agent는 할수 있는 action들 중에서 가장 action value가 높은 action을 선택합니다.

이 행동은 exploitation 입니다. 

<img src="https://www.dropbox.com/s/lfkl6sn8swtlpgu/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.025.jpeg?raw=1">



이렇게 선택된 action을 env.step()함수에 넣으면,  환경은 그 action으로 인한 다음 state, reward,done , info값을 리턴합니다. 

next state 는 그 action 으로 인해 변화되는 화면 이며, 

reward는 그 action으로 인한 reward 혹은 penalty값 입니다. 



done은 마리오가 죽었는지 살았는지에 대한 True, False 값이며 info는 디버깅을 하기 위한 정보가 들어있습니다. 



<img src="https://www.dropbox.com/s/p6erpqbmtminkzf/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.026.jpeg?raw=1">



그렇게 exploration과 explotation을 하며 마리오는 행동을 하게 되며 그 행동을 받은 슈퍼마리오 환경은 
위에서 설명드린 next state, reward, done, info값을 return합니다.



이 return된 값을 replay memory buffer 라고 불리우는 기억 창고 같은곳에 넣어서 학습을 하게 됩니다. 



<img src="https://www.dropbox.com/s/esu4v2cqw78v4et/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.027.jpeg?raw=1">



epsilon 값은 초기에는 1로 설정을 합니다. epsilon 값이 1 이라는 것은 100%의 확률로 임의의 action만 선택을 한다는 것이지요. 
<img src="https://www.dropbox.com/s/vvhmc5skq2elnb5/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.028.jpeg?raw=1">

1에서 시작된 epsilon 값은 최종적으로 0.1까지 줄이는데요 0.1의 의미는 10%의 확률로만 Exploration을 한다는 뜻입니다. 



<img src="https://www.dropbox.com/s/jlno8kqwx6lw2cx/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.029.jpeg?raw=1">

1에서 0.1까지 200000 time step으로 나눠서 점점 줄여나가는데요. 
이것은 초반에는 100%의 확률로 탐험을 하며 더 많은 경험을 해보고 학습을 하면 할수록 이 탐험을 하는 확률이 점점 줄어들게 됩니다. 

<img src="https://www.dropbox.com/s/ffo0f1bcar7eqru/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.030.jpeg?raw=1">

그렇게 선택된 action을 환경이 받고, 

<img src="https://www.dropbox.com/s/k5adzz0v7gy6or3/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.031.jpeg?raw=1">

환경이 준 정보들이 deque로 만든 replay memory buffer에 차곡차곡 쌓이게 됩니다. 





<img src="https://www.dropbox.com/s/cxhusa4ugm0g6yo/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.032.jpeg?raw=1">

매번 agent가 action을 선택하고 환경이 정보를  return해줄때마다 

 state, action, reward, next_state 값을 리플레이 메모리에 append  해줍니다. 

한계치 다다르면 새로운 정보가 들어왔을때 오래된 정보는 지워지고 새로운 정보가 들어옵니다. 



<img src="https://www.dropbox.com/s/wvksl1i632pr4kp/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.033.jpeg?raw=1">

이렇게 모여진 경험들을 기반으로 에이전트는 더 좋은 action을 선택하기 위해 학습을 합니다. 
그래서 더 좋은 action을 선택하기 위해서 신경망의 파라메터를 수정합니다. 

이렇게 파라메터를 수정하기 위해서 로스를 줄이는 과정이 필요합니다. 

<img src="https://www.dropbox.com/s/milxm78qdgi2eo4/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.034.jpeg?raw=1">

이 작업을 하기 위해 텐서플로우를 사용할게요. 

import tensorflow as tf로 텐서플로우를 임포트 합니다. 



<img src="https://www.dropbox.com/s/3i0v3i0k35t3rfr/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.035.jpeg?raw=1">



그래서 y값 ( target )값과 evaluation을 분리하고 이 둘의 차이를 줄이기 tf.reduce_mean으로 loss를 정의해줍니다. 

<img src="https://www.dropbox.com/s/lohqoemipq8y6tj/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.036.jpeg?raw=1">



Opitmizer는 AdamsOptimizer을 사용할 것이고요. 

<img src="https://www.dropbox.com/s/aywvdw1mykic4kj/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.037.jpeg?raw=1">

Optimizer을 사용해서 로스를 Minimize합니다. 

<img src="https://www.dropbox.com/s/zjrkby1fappwudv/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.038.jpeg?raw=1">



 이렇게 수정된 파라메터를 통하여 input으로 마리오 화면을 받고 Convolutional Neural Network를 통과합니다. 

이 네트워크를 통하여 output으로 나온 값은 슈퍼마리오 에이전트가 선태할수 있는 각 action의 action value 입니다. 

<img src="https://www.dropbox.com/s/m4lansc7up99irb/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.039.jpeg?raw=1">



전체적인 학습 구조를 보면, 

마리오 에이전트의 위치뿐만 아닌 움직임, 방향을 알기 위해 4 time step의 사진을 input으로 넣으면, 

Q-network( 위에서 설명한 Convolutional Neural network )를 통하여 파라메터에 의해 마리오가 선택할수 있는 각 action들의 action value가 output으로 return 됩니다. 
그 return된 action들중 exploitation 혹은 exploration 방법에 의해 action이 하나 선택되고,



그 선택된 action을 환경이 받아 그에 맞는 다음 state와 reward를 return합니다. 

state, action, reward, next state정보가 replay memory에 쌓이고 리플레이 메모리는 이 정보를 사용하여 Q-network의 파라메터를 슈퍼마리오 에이전트가 "목표를 달성하기 위해" 더 좋은 action 을 선택할 수 있도록 점점 수정합니다. 



마리오 는 이 과정을 몇 천번 거치면서 각 레벨 끝에 있는 깃발을 잡기 위해 순간순간 action을 선택합니다.



<img src="https://www.dropbox.com/s/7b4qvtrljbrs5z0/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.040.jpeg?raw=1">



<img src="https://www.dropbox.com/s/paqv98yilo5gt6y/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.041.jpeg?raw=1">



총 5000번 에피소드 정도 학습을 시켰고 이것은 4일 정도 소요되었습니다. 

<iframe width="790" height="444" src="https://www.youtube.com/embed/IjvbhwuCaF0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>







<img src="https://www.dropbox.com/s/hzahantdl7pjlpa/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.047.jpeg?raw=1">





이렇게 파이썬과 딥러닝 프레임워크 텐서플로우,케라스,파이토치만 있으면 강화학습을 이용하여 슈퍼마리오를 학습시킬수 있습니다. 



<img src="https://www.dropbox.com/s/kyt4dc6cmv8khfs/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.048.jpeg?raw=1">



다음 시리즈부터는 코드 리뷰 위주로 DQN부터 RainbowDQN 까지 총 여섯개로 나눠 공유해보겠습니다. 





<img src="https://www.dropbox.com/s/lvto9ux8n5n4x8s/%EC%B2%98%EC%9D%8C%EB%B6%80%ED%84%B0%EB%81%9D.049.jpeg?raw=1">








