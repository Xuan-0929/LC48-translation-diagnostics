"""Portable stdlib-only replay of exported machine artifacts; no semantic grades."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replay(root):
 manifest=read(root/'MANIFEST.json')
 for name,v in manifest['files'].items():
  rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
  p=root/rel;assert p.is_file() and not p.is_symlink();assert sha(p)==v['sha256'] and p.stat().st_size==v['bytes']
 sources=read(root/'data/sources60.json');sm={s['source_id']:s for s in sources};assert len(sm)==len(sources)==60
 rows=[json.loads(s) for s in (root/'data/generations568.jsonl').read_text().splitlines()];assert len(rows)==568
 assert len({r['record_id'] for r in rows})==568
 assert Counter(r['stage'] for r in rows)=={'original192':192,'factorial256':256,'numeral120':120}
 assets=read(root/'data/model_assets.json');conditions=Counter();unique_prompts=set();unique_contents=set()
 for r in rows:
  assert r['record_id']==r['stage']+':'+r['request_id'];s=sm[r['source_id']]
  assert r['source_text']==s['source_text'] and digest(r['source_text'])==s['source_sha256']==r['source_sha256']
  fields={k:r[k] for k in ['model_id','model_revision','source_id','source_sha256','prompt_condition','user_text','rendered_native_prompt','native_input_ids']}
  assert digest(json.dumps(fields,ensure_ascii=False,sort_keys=True,separators=(',',':')))==r['native_request_sha256']
  assert digest(r['rendered_native_prompt'])==r['rendered_prompt_sha256'];assert r['model_revision']==assets[r['model_id']]['model_revision']
  assert isinstance(r['native_input_ids'],list) and r['native_input_ids'] and all(type(x)is int and x>=0 for x in r['native_input_ids'])
  assert r['generation_status']=='completed' and r['generation_error'] is None
  tokens=r['generated_token_ids'];assert len(tokens)==r['generated_token_N']<=128 and all(type(x)is int and x>=0 for x in tokens)
  assert digest(r['translation_text'])==r['translation_sha256']
  assert r['ended_with_EOS']==bool(tokens and tokens[-1] in r['eos_token_ids'])
  assert r['empty_translation']==(not bool(r['translation_text'].strip()))
  assert r['reached_max_new_tokens']==(len(tokens)>=128);assert r['truncation_possible']==(not r['ended_with_EOS'] and len(tokens)>=128)
  conditions[r['stage'],r['model_id'],r['prompt_condition']]+=1
  unique_prompts.add((r['model_id'],tuple(r['native_input_ids'])));unique_contents.add((r['source_text'],r['translation_text']))
 # Anchor comparisons preserve repeated execution rather than calling it new sampling.
 lookup={(r['stage'],r['model_id'],r['source_id'],r['prompt_condition']):r for r in rows}
 def same(a,b):
  return all(a[k]==b[k] for k in ['source_text','user_text','rendered_native_prompt','native_input_ids','generated_token_ids','translation_text'])
 base64=[];word60=[]
 for r in rows:
  if r['stage']=='factorial256' and r['prompt_condition']=='I0B0':
   old=lookup['original192',r['model_id'],r['source_id'],'source_faithful'];assert same(old,r);base64.append(r['record_id'])
  if r['stage']=='numeral120' and sm[r['source_id']]['numeral_form']=='Chinese_words':
   src=sm[r['source_id']]['original_source_id'];p=r['prompt_condition']
   stage='original192' if p=='neutral' else 'factorial256'
   old=lookup[stage,r['model_id'],src,p];assert same(old,r);word60.append(r['record_id'])
 assert len(base64)==64 and len(word60)==60
 summary={'machine_execution_rows':568,'source_identifiers':60,'distinct_source_strings':len({s['source_text'] for s in sources}),'unique_model_native_input_sequences':len(unique_prompts),'distinct_exact_source_English_contents':len(unique_contents),'stage_rows':dict(Counter(r['stage'] for r in rows)),'baseline64_exact_native_input_English_and_token_anchors':len(base64),'numeral_word60_exact_native_input_English_and_token_anchors':len(word60),'EOS_rows':sum(r['ended_with_EOS'] for r in rows),'empty_rows':sum(r['empty_translation'] for r in rows),'possible_truncation_rows':sum(r['truncation_possible'] for r in rows),'conditions':[{'stage':a,'model':b,'condition':c,'rows':v} for (a,b,c),v in sorted(conditions.items())],'semantic_or_human_grades_included':0,'does_not_replay_primary_semantic_results':True,'GPU_regeneration_performed':False}
 assert summary['distinct_source_strings']==54 and summary['unique_model_native_input_sequences']==444
 expected=root/'expected/replay_summary.json'
 if expected.exists():assert summary==read(expected)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args();result=replay(ROOT)
 if a.out:a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
