from http.cookiejar import CookieJar
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor,Request,build_opener


def running_console_token_file():
    candidates=set()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            arguments=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if '18121' not in arguments or '--token-file' not in arguments:
                continue
            if not any(Path(argument).name=='rt_control_arm_console' for argument in arguments[:5]):
                continue
            candidate=Path(arguments[arguments.index('--token-file')+1])
            if candidate.is_file():
                candidates.add(candidate)
        except (PermissionError,FileNotFoundError,IndexError):
            continue
    if len(candidates)!=1:
        raise ValueError('不能唯一定位运行中的18121控制台令牌文件，请显式提供--console-token-file')
    return candidates.pop()


class ConsoleLease:
    def __init__(self,state,url,token_file):
        self.state=state
        self.url=url.rstrip('/')
        self.opener=build_opener(HTTPCookieProcessor(CookieJar()))
        self.request('/api/login',dict(token=Path(token_file).read_text().strip()))
        info=self.request('/api/info')
        joints={item['name']:item for item in info['joints']}
        from model import JOINT_NAMES
        if set(JOINT_NAMES)!=set(joints):
            raise ValueError('18121控制台必须恰好为左臂7轴')
        state.motor_bounds=[(joints[name]['lower'],joints[name]['upper']) for name in JOINT_NAMES]
        threading.Thread(target=self.watch,daemon=True).start()

    def request(self,path,data=None):
        body=None if data is None else json.dumps(data).encode()
        request=Request(self.url+path,data=body,headers={'Content-Type':'application/json','Origin':self.url})
        try:
            with self.opener.open(request,timeout=3) as response:
                return json.load(response)
        except HTTPError as error:
            try:message=json.load(error).get('message','控制台拒绝请求')
            except ValueError:message='控制台拒绝请求'
            raise ValueError(message) from None

    def watch(self):
        while True:
            try:
                with self.opener.open(self.url+'/api/events',timeout=3) as response:
                    for line in response:
                        if not line.startswith(b'data: '):
                            continue
                        status=json.loads(line[6:])
                        with self.state.lock:
                            self.state.console_status=status
                            self.state.console_received=time.monotonic()
                        if status.get('lease')=='you' and time.monotonic()-self.state.viewer_seen>5:
                            if self.state.active:
                                if self.state.active['direction']!='stop':
                                    self.state.stop()
                            else:
                                self.request('/api/release',{})
            except Exception:
                with self.state.lock:
                    self.state.console_received=0.
                time.sleep(.5)

    def acquire(self):
        return self.request('/api/acquire',{})

    def release(self):
        return self.request('/api/release',{})
