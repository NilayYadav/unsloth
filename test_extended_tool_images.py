"""Extra black-box conversion/admission cases for the review evidence matrix."""
import copy
import sys
from pathlib import Path
repo=Path(__file__).resolve().parents[1]/'wt_r10762'
sys.path[:0]=[str(repo/'studio/backend'),str(repo)]
import pytest
from core.inference.anthropic_compat import anthropic_messages_to_openai,fold_tool_results_into_user
from models.inference import AnthropicMessagesRequest
from routes import inference as inf
from studio.backend.tests.test_anthropic_native_tool_images import image_block,payload

@pytest.mark.parametrize('source_type',['base64','data_url','https_url'])
@pytest.mark.parametrize('history',[False,True])
def test_mixed_tool_result_filters_unknown_parts_without_losing_images(source_type,history):
 block=image_block()
 if source_type!='base64':
  block['source']={'type':'url','url':('data:image/webp;base64,'+block['source']['data']) if source_type=='data_url' else 'https://example.com/capture.png'}
 parts=[None,'ignored',{'type':'future_block','data':'ignored'},{'type':'text','text':'start'},block,copy.deepcopy(block),{'type':'text','text':'end'}]
 body=payload(parts)
 if history:body['messages'] += [{'role':'assistant','content':'Previous inspection.'},{'role':'user','content':'Inspect it again.'}]
 native=AnthropicMessagesRequest(**body); original=copy.deepcopy(body)
 wire=anthropic_messages_to_openai([m.model_dump() for m in native.messages])
 result=wire[2]['content']
 assert [p['type'] for p in result]==['text','image_url','image_url','text']
 assert result[0]['text']=='start' and result[-1]['text']=='end'
 folded=fold_tool_results_into_user(wire)
 assert folded[2]['content'][1:]==result
 compact,count=inf._openai_llama_admission_messages_for_estimate(native.messages)
 assert count==2
 assert inf._anthropic_request_has_image(native)
 assert len(inf._anthropic_local_image_payloads(native))==(0 if source_type=='https_url' else 2)
 assert body==original
 assert len(str(compact))<3000

def test_nonforwardable_image_does_not_erase_adjacent_tool_text():
 body=payload([{'type':'text','text':'start'},{'type':'image','source':{'type':'url'}},{'type':'text','text':'end'}])
 native=AnthropicMessagesRequest(**body)
 wire=anthropic_messages_to_openai([m.model_dump() for m in native.messages])
 assert wire[2]['content']=='start end'
 assert wire[2]['tool_call_id']=='toolu_first'
