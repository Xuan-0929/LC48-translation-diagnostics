"""Optional actual native encoding/decoding check with locally supplied pinned assets."""
from pathlib import Path
import argparse,json
from collections import Counter
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--qwen-path',type=Path,required=True);p.add_argument('--mistral-path',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 rows=[json.loads(s) for s in (ROOT/'data/generations568.jsonl').read_text().splitlines()];assets=json.loads((ROOT/'data/model_assets.json').read_text());counts=Counter()
 for mid,path in [('Qwen/Qwen3-8B',a.qwen_path),('mistralai/Mistral-7B-Instruct-v0.3',a.mistral_path)]:
  import hashlib
  for f in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','config.json']:
   expected=assets[mid]['file_sha256'].get(f)
   if expected is not None:assert hashlib.sha256((path/f).read_bytes()).hexdigest()==expected
  tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
  for r in rows:
   if r['model_id']!=mid:continue
   kw={'add_generation_prompt':True}
   if mid.startswith('Qwen/'):kw['enable_thinking']=False
   messages=[{'role':'user','content':r['user_text']}]
   assert tokenizer.apply_chat_template(messages,tokenize=False,**kw)==r['rendered_native_prompt']
   assert tokenizer.apply_chat_template(messages,tokenize=True,**kw)==r['native_input_ids']
   assert tokenizer.decode(r['generated_token_ids'],skip_special_tokens=False)==r['decoded_with_special_tokens']
   assert tokenizer.decode(r['generated_token_ids'],skip_special_tokens=True)==r['translation_text'];counts[mid]+=1
 result={'all568_native_prompt_texts_input_ids_and_two_output_decodes_match':sum(counts.values())==568,'rows_by_model':dict(counts),'used_local_assets_only':True,'GPU_or_generation_performed':False,'not_a_semantic_grade_validation':True};assert sum(counts.values())==568;a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
