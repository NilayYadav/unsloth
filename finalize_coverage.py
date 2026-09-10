import json,subprocess,re
from pathlib import Path
root=Path(__file__).resolve().parent;c=json.loads((root/'coverage.json').read_text());out={}
for name,e in c['files'].items():
 diff=subprocess.check_output(['git','-C',str(root.parent/'wt_r10762'),'diff','--unified=0','d0dbe905','--',name],text=True)
 added=set();line=0
 for l in diff.splitlines():
  m=re.match(r'@@ .* \+(\d+)',l)
  if m:line=int(m[1]);continue
  if l.startswith('+++'):continue
  if l.startswith('+'):added.add(line);line+=1
  elif l.startswith(' '):line+=1
 ex=set(e['executed_lines']);missing=set(e['missing_lines']);changed=added&(ex|missing)
 branches=[b for b in e['executed_branches']+e['missing_branches'] if b[0] in changed]
 missed=[b for b in e['missing_branches'] if b[0] in changed]
 out[name]={'changed_executable_lines':len(changed),'covered':len(changed&ex),'missing':sorted(changed&missing),'branches_from_changed_lines':len(branches),'missing_branches':missed,'file_statement_coverage':e['summary']['percent_statements_covered']}
 assert not out[name]['missing'] and not missed,out[name]
print(json.dumps(out,indent=2));(root/'changed-coverage.json').write_text(json.dumps(out,indent=2))
