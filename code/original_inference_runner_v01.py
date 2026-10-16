"""Two pinned unadapted models, fixed native requests, bounded inference only.

Admission requires a fresh platform quote and saved shutdown timer. This script
does not boot an instance, set its timer, download weights, or invent those facts.
Every started generation is retained, including empty/error/truncated outputs.
"""
import argparse, gc, hashlib, json, math, os, random, signal, time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

PINS = {'Qwen/Qwen3-8B':'b968826d9c46dd6066d109eabc6255188de91218',
        'mistralai/Mistral-7B-Instruct-v0.3':'c170c708c41dac9275d15a8fff4eca08d52bab71'}
MODEL_ORDER = list(PINS)
REQUIRED_RUNTIME = {'torch':'2.5.1+cu124','transformers':'4.51.3','cuda':'12.4'}

def require(v, msg):
 if not v: raise ValueError(msg)
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def epoch(s):
 d=datetime.fromisoformat(s.replace('Z','+00:00'))
 require(d.tzinfo is not None,'timezone required'); return d.timestamp()
def atomic(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def validate_requests(requests, release):
 require(release.get('schema_version')=='LC48_qualified_source_release_v01' and
   release.get('status')=='released_for_inference' and release.get('source_contract_qualification_complete') is True,
   'source qualification incomplete')
 source_hashes=release['source_sha256_by_id']
 require(len(source_hashes)==48,'full frozen48 source set required')
 seen=set()
 for r in requests:
  key=(r['model_id'],r['prompt_condition'],r['source_id'])
  require(key not in seen,'duplicate request');seen.add(key)
  require(r['model_revision']==PINS.get(r['model_id']),'model revision changed')
  require(r['source_sha256']==source_hashes.get(r['source_id']),'source identity changed')
  require(r.get('candidate_only') is False and r.get('released_for_GPU_execution') is True,'candidate not released')
  fields={k:r[k] for k in ['model_id','model_revision','source_id','source_sha256','prompt_condition','user_text','rendered_native_prompt','native_input_ids']}
  require(r['native_request_sha256']==digest(json.dumps(fields,ensure_ascii=False,sort_keys=True,separators=(',',':'))),'native request drift')
  require(r['rendered_prompt_sha256']==digest(r['rendered_native_prompt']),'native rendering drift')
  require(isinstance(r['native_input_ids'],list) and bool(r['native_input_ids']) and
   all(type(i) is int and i>=0 for i in r['native_input_ids']),'invalid input IDs')
 require(seen=={(m,p,s) for m in PINS for p in ['neutral','source_faithful'] for s in source_hashes},'incomplete request grid')

def check_admission(lock, bundle, now=None, *, engineering=False):
 now=time.time() if now is None else now
 require(lock.get('schema_version')=='LC48_actual_paid_boot_admission_v01' and lock.get('authorized') is True,'actual admission missing')
 require(engineering or not lock.get('engineering_fixture'),'engineering fixture cannot admit paid execution')
 require(lock.get('per_boot_cap_cny')==15 and lock.get('aggregate_cap_cny') is None,'per boot budget must remain15, no aggregate cap')
 require(lock.get('required_runtime')==REQUIRED_RUNTIME,'runtime changed')
 require(set(lock['input_sha256'])=={'run_LC48_native_inference_v01.py','requests.json','source_release.json','pinned_model_assets.json'},'complete frozen input set required')
 for n,h in lock['input_sha256'].items():
  p=Path(n);require(not p.is_absolute() and '..' not in p.parts,'unsafe input path')
  require(sha(Path(bundle)/p)==h,'frozen bundle drift: '+n)
 require(lock.get('official_quote_observed') is True and lock.get('platform_timer_saved_and_verified') is True,
   'platform quote and saved timer not verified')
 for stem in ['official_quote','official_timer']:
  require(sha(lock[stem+'_path'])==lock[stem+'_sha256'],'platform evidence drift')
 quote=read(lock['official_quote_path']);timer=read(lock['official_timer_path'])
 require(engineering or not (quote.get('engineering_fixture') or timer.get('engineering_fixture')),'engineering platform evidence cannot admit paid execution')
 require(quote['instance_id']==timer['instance_id']==lock['instance_id'],'different instance')
 require(quote.get('current_GPU_hourly_rate_confirmed') is True and timer.get('saved_platform_shutdown_confirmed') is True,'unconfirmed platform observation')
 rate=Decimal(str(quote['rate_cny_per_hour']));fixed=Decimal(str(quote['fixed_fee_cny']));minimum=Decimal(str(quote['minimum_charge_cny']))
 require(rate.is_finite() and rate>0 and fixed.is_finite() and fixed>=0 and minimum.is_finite() and minimum>=0,'invalid quote')
 reserve=Decimal(str(lock['fee_reserve_cny']));require(reserve.is_finite() and reserve>=Decimal('.5'),'fee reserve must be at least .5')
 boot=epoch(lock['boot_bound_start_utc']);shutdown=epoch(timer['shutdown_utc']);deadline=epoch(lock['runner_deadline_utc'])
 require(0<=now-epoch(quote['observed_utc'])<=1800,'quote not freshly checked')
 require(boot<=now<deadline and shutdown-deadline>=300,'missing five minute shutdown reserve')
 require(boot<=epoch(quote['observed_utc'])<=now,'quote does not belong to this boot window')
 require(epoch(lock['platform_shutdown_utc'])==shutdown,'shutdown evidence inconsistent')
 require(quote.get('charge_rounding')=='whole_hour_upper_bound','whole hour conservative charge bound required')
 hours=(Decimal(str(shutdown-boot))/Decimal(3600)).to_integral_value(rounding=ROUND_CEILING)
 worst=fixed+max(minimum,rate*hours)+reserve
 require(worst<=Decimal('15'),'worst per-boot cost exceeds15')
 return dict(runner_deadline_epoch=deadline,worst_bound_cost_cny=str(worst),shutdown_epoch=shutdown,billing_hours_upper_bound=str(hours))

def verify_model_files(path, spec, tick=lambda:None):
 path=Path(path);verified={}
 for n,h in spec['file_sha256'].items():
  p=Path(n);require(not p.is_absolute() and '..' not in p.parts,'unsafe model asset path')
  tick();require(sha(path/p)==h,'pinned model file differs: '+n);verified[n]=h
 index=read(path/'model.safetensors.index.json')
 actual=set(index['weight_map'].values())
 require(actual==set(spec['weight_file_names']),'weight index/shard manifest differs')
 require(actual<=set(verified),'not all weights verified')
 require(spec['model_revision']==PINS.get(spec['model_id']),'weight revision changed')
 return verified

def output_record(r, token_ids, tokenizer, eos_ids, raw_path, status='completed', error=None):
 raw=tokenizer.decode(token_ids,skip_special_tokens=False)
 translation=tokenizer.decode(token_ids,skip_special_tokens=True)
 ended=bool(token_ids) and token_ids[-1] in eos_ids
 return dict(request_id=r['request_id'],model_id=r['model_id'],model_revision=r['model_revision'],
  prompt_condition=r['prompt_condition'],source_id=r['source_id'],source_sha256=r['source_sha256'],
  native_request_sha256=r['native_request_sha256'],generation_provenance_kind='actual_model_execution',
  raw_generation_artifact_path=str(raw_path),generation_status=status,generation_error=error,
  generated_token_ids=token_ids,decoded_with_special_tokens=raw,
  translation_text=translation,translation_sha256=digest(translation),
  generated_token_N=len(token_ids),eos_token_ids=eos_ids,ended_with_EOS=ended,
  reached_max_new_tokens=len(token_ids)>=128,empty_translation=not bool(translation.strip()),
  truncation_possible=not ended and len(token_ids)>=128,
  language_validity_not_automatically_inferred=True,completed_utc=utc())

def run(args):
 bundle=args.bundle.resolve();lock=read(args.admission)
 admission=check_admission(lock,bundle)
 requests=read(bundle/'requests.json');release=read(bundle/'source_release.json')
 validate_requests(requests,release)
 require(all(r['source_release_sha256']==sha(bundle/'source_release.json') for r in requests),'release binding differs')
 args.out.mkdir(parents=True,exist_ok=False)
 report=dict(schema_version='LC48_actual_native_inference_report_v01',started_utc=utc(),
  status='running',planned_requests_N=192,records=[],models=[],optimizer_updates=0,
  execution_admission_sha256=sha(args.admission),admission_budget_bound=admission)
 def save(): atomic(args.out/'actual_generation_report.json',report)
 def tick():
  if time.time()>=admission['runner_deadline_epoch']: raise TimeoutError('paid boot runner deadline')
 def alarm(*_): raise TimeoutError('paid boot runner deadline')
 signal.signal(signal.SIGALRM,alarm)
 signal.alarm(max(1,math.floor(admission['runner_deadline_epoch']-time.time())))
 save()
 try:
  require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in [None,':4096:8'],'conflicting cuBLAS environment')
  os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
  import torch,transformers
  from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
  versions=dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda)
  require(versions==REQUIRED_RUNTIME,'actual runtime differs: '+str(versions))
  require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA BF16 required')
  torch.use_deterministic_algorithms(True,warn_only=False)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  torch.backends.cuda.matmul.allow_tf32=False
  torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
  torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  torch.set_grad_enabled(False);torch.manual_seed(20261008);random.seed(20261008)
  report['actual_runtime']=versions
  report['gpu_name']=torch.cuda.get_device_name(0)
  manifest=read(bundle/'pinned_model_assets.json')
  for mid in MODEL_ORDER:
   tick();spec=manifest[mid];model_path=Path(lock['model_paths'][mid])
   verified=verify_model_files(model_path,spec,tick)
   tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
   rows=[r for r in requests if r['model_id']==mid]
   random.Random(20261008).shuffle(rows)
   for r in rows:
    kw=dict(add_generation_prompt=True)
    if mid=='Qwen/Qwen3-8B':kw['enable_thinking']=False
    messages=[dict(role='user',content=r['user_text'])]
    require(tokenizer.apply_chat_template(messages,tokenize=False,**kw)==r['rendered_native_prompt'],'native template changed')
    require(tokenizer.apply_chat_template(messages,tokenize=True,**kw)==r['native_input_ids'],'native input changed')
   native_gc=GenerationConfig.from_pretrained(model_path,local_files_only=True)
   eos=native_gc.eos_token_id if isinstance(native_gc.eos_token_id,list) else [native_gc.eos_token_id]
   require(bool(eos) and all(type(i) is int for i in eos),'native EOS IDs missing')
   pad=native_gc.pad_token_id if native_gc.pad_token_id is not None else tokenizer.eos_token_id
   gc_config=GenerationConfig(do_sample=False,max_new_tokens=128,num_beams=1,
    temperature=1.0,top_p=1.0,top_k=0,min_p=None,typical_p=1.0,
    epsilon_cutoff=0.0,eta_cutoff=0.0,repetition_penalty=1.0,
    eos_token_id=eos,pad_token_id=pad,use_cache=True)
   tick();model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
     torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'})
   model.eval();model.requires_grad_(False)
   report['models'].append(dict(model_id=mid,revision=PINS[mid],verified_asset_sha256=verified,
    actual_dtype=str(next(model.parameters()).dtype),actual_attention_backend=model.config._attn_implementation,
    native_generation_config=native_gc.to_dict(),actual_generation_config=gc_config.to_dict(),
    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),trainable_parameters=0))
   save()
   for r in rows:
    tick();start=utc();raw_path=args.out/('request-'+r['request_id']+'.json')
    atomic(raw_path,dict(request=r,status='attempt_started',started_utc=start))
    tokens=[];status='completed';error=None
    try:
     ids=torch.tensor([r['native_input_ids']],device='cuda:0',dtype=torch.long)
     with torch.inference_mode():
      generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
        generation_config=gc_config,use_model_defaults=False)
     tokens=generated[0,len(r['native_input_ids']):].detach().cpu().tolist()
     del generated,ids
    except TimeoutError as e:
     status='interrupted';error=repr(e)
    except Exception as e:
     status='failed';error=repr(e)
    record=output_record(r,tokens,tokenizer,eos,raw_path,status,error)
    atomic(raw_path,dict(request=r,status=status,started_utc=start,output=record))
    report['records'].append(record);save()
    print(json.dumps(dict(completed_attempts_N=len(report['records']),model_id=mid,status=status)),flush=True)
    if status!='completed': raise RuntimeError('generation attempt preserved; no automatic retry: '+str(error))
   del model,tokenizer;gc.collect();torch.cuda.empty_cache()
  require(len(report['records'])==192,'request grid incomplete')
  report['status']='complete_pending_local_provenance_intake_and_human_review'
 except BaseException as e:
  report['status']='partial_failure_preserved';report['terminal_error']=repr(e)
  raise
 finally:
  signal.alarm(0);report['finished_utc']=utc();save()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True)
 ap.add_argument('--admission',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 run(ap.parse_args())
