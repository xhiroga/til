# Rye

```console
% rye init my-project
success: Initialized project in /Users/hiroga/.ghq/github.com/xhiroga/til/software-engineering/python/rye/_src/basics/my-project
  Run `rye sync` to get started

% rye pin 3.10 
pinned 3.10.11 in /Users/hiroga/.ghq/github.com/xhiroga/til/software-engineering/python/rye/_src/my-project/.python-version

% rye sync    
Python version mismatch (found cpython@3.11.3, expect cpython@3.10.11), recreating.
Initializing new virtualenv in /Users/hiroga/.ghq/github.com/xhiroga/til/software-engineering/python/rye/_src/my-project/.venv
Python version: cpython@3.10.11
Generating production lockfile: /Users/hiroga/.ghq/github.com/xhiroga/til/software-engineering/python/rye/_src/my-project/requirements.lock
Creating virtualenv for pip-tools
Generating dev lockfile: /Users/hiroga/.ghq/github.com/xhiroga/til/software-engineering/python/rye/_src/my-project/requirements-dev.lock
Installing dependencies
Looking in indexes: https://pypi.org/simple/
Obtaining file:///. (from -r /var/folders/wp/s8z2z9pd0f9bkf8_h4w0ptdm0000gp/T/tmpxuak0p0f (line 1))
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Building wheels for collected packages: my-project
  Building editable for my-project (pyproject.toml) ... done
  Created wheel for my-project: filename=my_project-0.1.0-py3-none-any.whl size=1238 sha256=f05acf3116e022a0b302bd3b3d9873ecca0a763253c4dc40a56eb716ee81d05f
  Stored in directory: /private/var/folders/wp/s8z2z9pd0f9bkf8_h4w0ptdm0000gp/T/pip-ephem-wheel-cache-51giaoz7/wheels/78/12/b9/db4c72ae1aebdfd866f484b59eb7b39f206b7d7d1080ae28b1
Successfully built my-project
Installing collected packages: my-project
Successfully installed my-project-0.1.0
Done!
```

## References

- [Basics - Rye](https://rye-up.com/guide/basics/)
