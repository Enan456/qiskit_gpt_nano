# Training Results


## Configuration

- n_embd: 16
- n_head: 2
- n_layer: 2
- block_size: 8
- batch_size: 4
- learning_rate: 0.001
- device: cpu
- max_iters: 1000
- eval_iters: 20


## Results Summary

- Classical loss: 2.7006 (wall: 2.8s, cpu: 4.3s, rss_max: 358.9 MB)
- Quantum loss: 2.7581 (wall: 1102.9s, cpu: 2836.6s, rss_max: 415.2 MB)
- Classical params: 8,769
- Quantum params: 8,821
- Classical throughput: 11281.5 tokens/s
- Quantum throughput: 29.0 tokens/s

## Output Difference Stats

- Hamming distance (tokens, first 121): 101
- Agreement ratio: 0.165
- Jaccard overlap (token sets): 0.375
- Levenshtein distance (chars): 26

## Sample Outputs

### Classical GPT

```

I
Te he be the the the te the and the the the the the the the the the the the anou the the t the me the the the the anou
```

### Quantum GPT

```


An he he the the the thethe the the the thethe the the the the he the the the the these the the the thes t the the the 
```

## Profiling (Top 25 functions)

### Classical

```
         3050688 function calls (2875312 primitive calls) in 3.372 seconds

   Ordered by: cumulative time
   List reduced from 4389 to 25 due to restriction <25>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    24/18    0.000    0.000    3.691    0.205 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/_ops.py:369(fallthrough)
      2/1    0.000    0.000    2.826    2.826 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/threading.py:641(wait)
        1    0.000    0.000    2.826    2.826 /Users/enansrivastava/quantum/train.py:169(<lambda>)
        1    0.048    0.048    2.826    2.826 /Users/enansrivastava/quantum/train.py:59(train_model)
       20    0.000    0.000    1.279    0.064 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/_ops.py:307(py_impl)
     1000    0.001    0.000    1.059    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/_tensor.py:592(backward)
     1000    0.003    0.000    1.058    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/autograd/__init__.py:243(backward)
     1000    0.001    0.000    1.047    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/autograd/graph.py:820(_engine_run_backward)
     1000    1.045    0.001    1.045    0.001 {method 'run_backward' of 'torch._C._EngineBase' objects}
70000/1400    0.029    0.000    1.030    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/nn/modules/module.py:1769(_wrapped_call_impl)
70000/1400    0.053    0.000    1.029    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/nn/modules/module.py:1777(_call_impl)
     1400    0.009    0.000    1.026    0.001 /Users/enansrivastava/quantum/src/quantum_gpt_implementation.py:289(forward)
4200/1400    0.008    0.000    0.881    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/nn/modules/container.py:242(forward)
     2800    0.015    0.000    0.876    0.000 /Users/enansrivastava/quantum/src/quantum_gpt_implementation.py:272(forward)
        3    0.000    0.000    0.845    0.282 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/_ops.py:172(py_functionalize_impl)
     1000    0.005    0.000    0.566    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/optim/optimizer.py:496(wrapper)
     2800    0.010    0.000    0.562    0.000 /Users/enansrivastava/quantum/src/quantum_gpt_implementation.py:255(forward)
     1000    0.003    0.000    0.539    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/optim/optimizer.py:61(_use_grad)
     1000    0.002    0.000    0.534    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/optim/adam.py:213(step)
     5600    0.155    0.000    0.491    0.000 /Users/enansrivastava/quantum/src/quantum_gpt_implementation.py:219(forward)
     1000    0.001    0.000    0.474    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/optim/optimizer.py:132(maybe_fallback)
     1000    0.002    0.000    0.472    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/optim/adam.py:881(adam)
     1000    0.178    0.000    0.458    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/optim/adam.py:345(_single_tensor_adam)
        3    0.000    0.000    0.435    0.145 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/_ops.py:279(__init__)
        1    0.000    0.000    0.410    0.410 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/_higher_order_ops/triton_kernel_wrap.py:987(__init__)



```

### Quantum

```
         3156981177 function calls (3106045007 primitive calls) in 1090.981 seconds

   Ordered by: cumulative time
   List reduced from 4520 to 25 due to restriction <25>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   153600  361.499    0.002  361.499    0.002 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/backends/backend_utils.py:437(cpp_execute_circuits)
    64000    0.062    0.000  355.436    0.006 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_machine_learning/gradients/base/base_estimator_gradient.py:104(run)
    64000    0.168    0.000  355.201    0.006 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_machine_learning/algorithm_job.py:28(submit)
   217601    0.639    0.000  315.546    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/threading.py:955(start)
  1625608    6.198    0.000  273.656    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/primitives/sampler.py:111(_circuit_key)
 63244976   56.278    0.000  264.749    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/primitives/sampler.py:128(<genexpr>)
123238736   36.206    0.000  164.804    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/primitives/sampler.py:91(_bits_key)
209971792   50.200    0.000  128.598    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/primitives/sampler.py:92(<genexpr>)
217600/159919    0.461    0.000   88.421    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit/primitives/primitive_job.py:41(_submit)
  1625600   12.103    0.000   65.688    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/primitives/estimator.py:614(run)
   153600    0.147    0.000   53.503    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit/primitives/base/base_estimator.py:121(run)
   153600    0.291    0.000   53.356    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/primitives/estimator.py:223(_run)
175514112   32.726    0.000   47.582    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit/circuit/quantumcircuit.py:3125(find_bit)
478372726/478368021   26.651    0.000   35.801    0.000 {built-in method builtins.isinstance}
   153600    0.286    0.000   33.270    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit/primitives/base/validation.py:35(_validate_estimator_args)
173466112   23.712    0.000   31.609    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/primitives/sampler.py:95(<genexpr>)
    64000    0.498    0.000   30.618    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_machine_learning/gradients/base/base_estimator_gradient.py:180(_preprocess)
 93773112    7.932    0.000   30.509    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/primitives/sampler.py:133(<genexpr>)
 23526416    6.461    0.000   29.941    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit/utils/deprecation.py:94(wrapper)
  1625600   12.348    0.000   29.631    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/primitives/estimator.py:665(_pauli_expval_with_variance)
   153600    0.619    0.000   27.566    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit/primitives/base/validation.py:130(_validate_parameter_values)
   153600    0.193    0.000   27.416    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/backends/aerbackend.py:128(_convert_binds)
   153600    4.461    0.000   27.186    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/backends/aerbackend.py:96(_convert_circuit_binds)
   153600   14.763    0.000   23.658    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/primitives/estimator.py:348(_pre_process_params)
   153600    4.591    0.000   23.322    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/qiskit_aer/primitives/estimator.py:434(_create_post_processing)



```
