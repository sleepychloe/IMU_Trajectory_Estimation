- goal of this experiment:
튜닝 비용 절감을 위한 segment-based proxy objective의 타당성 검증



- why?
긴 데이터에서는 전체 실험3 튜닝 비용이 너무 큼,전체 데이터를 쓰는 방식이 비실용적
-> 시간은 얼마나 줄어드는지, 최종 full-data 성능은 얼마나 유지되는지 확인



- seg
seg1: head 5s + trim 3s every 15s + tail 5s
seg2: head 10s + trim 5s every 15s + tail 10s



- 형태
exp3: full tuning → full eval
exp4: seg tuning → full eval

in each exp(4-1, ..., 4-6)
	same search space
	same n_trials
	same seed
	same data
	eval: always full data

measurement:
	min/max/mean/p90(in rad, in deg)
	running time
	speedup
	best parameter difference



res: data 1-data3
- 실험 3과 실험 4의 결과 경향이 거의 동일 (어떤 계열이 좋은지, gating이 없는 것보다 fixed norm + mag innov 쪽이 좋은지, time-varying 계열은 상대적으로 불리한지)
- SEG 기반 튜닝을 써도 full-data 성능이 크게 무너지지 않음
- 경우에 따라서 오히려 더 좋아질 때도 있음을 관찰
- 시간 절감 효과 매우 큼
- data 1,2: exp 4가 더 좋거나 거의 동일
- data 3: best가 바뀌었지만 원래 exp3-3/3-4차이 미미했음
- 두 SEG 설정 중에서는 SEG2(5s window) 가 더 안정적으로 좋은 결과를 보임



- why improved?
1. full-data objective가 항상 더 좋은 objective는 아님

전체 시퀀스는 너무 긴 쉬운 구간/비슷한 motion이 오래 반복되는 구간/특정 disturbance가 길게 이어지는 구간/
실전 generalization엔 덜 중요한 구간 등을 포함.
SEG는 시간축에 퍼진 대표 구간만 확인, 긴 구간의 편향을 줄여서 더 좋은 파라미터가 나올 수 있음

2. 일종의 regularization 효과

SEG tuning은 전체 데이터에 대한 exact fitting
보다 대표 구간에 대한 proxy fitting이므로
full-data tuning이 데이터셋의 특정 구간 패턴에 약간 끌려가던 걸 SEG가 덜 끌려가게 만들 수 있음

3. objective는 recursive system이라 long-horizon effect가 큼

어떤 파라미터는 초반엔 좋고 후반엔 나쁘거나, 특정 drift pattern에만 유리하거나, 특정 disturbance에만 유리할 수 있음.
SEG는 각 구간을 끊어서 봐서 long-horizon accumulated artifact의 영향을 줄여줌, 지역적으로 안정적인 파라미터가 더 잘 뽑힐 수 있음

4. segment reset 자체가 local behavior를 더 잘 보게 해줌

segmented scorer는 각 구간에서 q0 = q_ref[seg_s] 로 reset후 local orientation quality를 평가.
full-data tuning은 긴 드리프트 누적까지 포함한 전체 trajectory 최적화이고 SEG tuning은 여러 대표 구간에서 local fusion quality 최적화에 가까움

5. finite trial에서는 objective가 단순한 쪽이 유리할 수 있음

Optuna trial이 무한대가 아닌 비교적 적은 횟수라면 문제가 복잡할수록 최적점 근처를 못 찾을 수 있음.
SEG objective는 보통 더 단순하고, 더 noisy하지 않고, basin이 더 찾기 쉬울 수 있음.
즉 연산량이 적어서 정확도가 올라감은 아니고 연산량이 적어지면서 더 다루기 쉬운 objective가 되어 제한된 trial 안에서 더 좋은 해를 찾음에 가까움



- conclusion (after running data4):

FULL DATA는 의미 없다X
SEG가 더 우수하다X

Full-data tuning은 개념적으로 가장 직접적인 접근이지만, 긴 시계열에서는 계산 비용이 과도하게 증가하였다. 반면 segment-based tuning은 실행 시간을 크게 줄이면서도 full-sequence 기준에서 유사한 정확도를 유지했고, 일부 경우에는 오히려 더 나은 결과를 보였다. 따라서 이후 실험에서는 full-data tuning의 지속 사용을 정당화할 만큼의 충분한 실용적 이점을 관찰하지 못하였다.

(Full-data tuning is conceptually the most direct approach, but its computational cost became prohibitive for long sequences. Since segment-based tuning substantially reduced runtime while preserving comparable full-sequence accuracy, and occasionally even improving it, we did not observe enough practical benefit to justify continued use of full-data tuning in subsequent experiments.)


