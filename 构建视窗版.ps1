param(
    [string]$Python解释器 = 'python',
    [switch]$跳过依赖安装,
    [switch]$跳过工具链安装,
    [switch]$跳过冒烟测试
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function 输出步骤 {
    param([string]$消息)
    Write-Host "`n==> $消息" -ForegroundColor Cyan
}

function 调用并检查 {
    param(
        [string]$文件路径,
        [string[]]$参数列表 = @(),
        [string]$工作目录 = $PWD.Path,
        [string]$失败提示 = '命令执行失败'
    )

    Push-Location $工作目录
    try {
        & $文件路径 @参数列表
        if ($LASTEXITCODE -ne 0) {
            throw "$失败提示 (退出码: $LASTEXITCODE)"
        }
    }
    finally {
        Pop-Location
    }
}

function 获取Python信息 {
    param([string]$Python命令)

    $查询脚本 = @'
import json
import site
import sys

信息 = {
    "executable": sys.executable,
    "version": list(sys.version_info[:3]),
    "user_site": site.getusersitepackages(),
}
print(json.dumps(信息))
'@

    $结果文本 = & $Python命令 -c $查询脚本
    if ($LASTEXITCODE -ne 0) {
        throw '无法读取 Python 环境信息'
    }
    return $结果文本 | ConvertFrom-Json
}

function 确保Python包 {
    param(
        [string]$Python命令,
        [string]$导入名,
        [string]$包名 = $导入名
    )

    & $Python命令 -c "import $导入名" *> $null
    if ($LASTEXITCODE -eq 0) {
        return
    }

    输出步骤 "安装用户级 Python 包: $包名"
    调用并检查 -文件路径 $Python命令 -参数列表 @('-m', 'pip', 'install', '--user', $包名) -失败提示 "安装 $包名 失败"
}

function 查找VsWhere {
    $查找器路径 = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $查找器路径) {
        return $查找器路径
    }
    return $null
}

function 查找VcVars64 {
    $查找器路径 = 查找VsWhere
    if ($查找器路径) {
        $候选安装路径列表 = @(
            (& $查找器路径 -prerelease -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath),
            (& $查找器路径 -prerelease -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath),
            (& $查找器路径 -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath),
            (& $查找器路径 -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
        ) | Where-Object { $_ }

        foreach ($安装路径 in $候选安装路径列表) {
            $初始化脚本 = Join-Path $安装路径 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path $初始化脚本) {
                return $初始化脚本
            }
        }
    }

    $固定候选路径列表 = @(
        'C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    )
    foreach ($初始化脚本 in $固定候选路径列表) {
        if (Test-Path $初始化脚本) {
            return $初始化脚本
        }
    }

    return $null
}

function 安装构建工具 {
    if ($跳过工具链安装) {
        throw '未找到 MSVC 构建工具，且已禁止自动安装'
    }

    输出步骤 '通过 winget 安装 Visual Studio Build Tools'
    $安装参数 = '--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
    调用并检查 -文件路径 'winget' -参数列表 @(
        'install', '--id', 'Microsoft.VisualStudio.2022.BuildTools', '--exact', '--silent',
        '--accept-package-agreements', '--accept-source-agreements', '--override', $安装参数
    ) -失败提示 '安装 Visual Studio Build Tools 失败'
}

function 调用Msvc批处理 {
    param(
        [string]$VcVars路径,
        [string]$工作目录,
        [string[]]$命令列表,
        [string]$失败提示
    )

    $临时脚本 = Join-Path $env:TEMP ("lucaschess_build_{0}.cmd" -f ([guid]::NewGuid().ToString('N')))
    $脚本内容 = @(
        '@echo off',
        'setlocal',
        "call `"$VcVars路径`"",
        'if errorlevel 1 exit /b 1',
        "cd /d `"$工作目录`""
    ) + $命令列表
    Set-Content -Path $临时脚本 -Value $脚本内容 -Encoding ASCII

    try {
        调用并检查 -文件路径 'cmd.exe' -参数列表 @('/d', '/c', $临时脚本) -失败提示 $失败提示
    }
    finally {
        Remove-Item $临时脚本 -ErrorAction SilentlyContinue
    }
}

$仓库根目录 = Split-Path -Parent $MyInvocation.MyCommand.Path
$二进制目录 = Join-Path $仓库根目录 'bin'
$快码目录 = Join-Path $二进制目录 '_fastercode'
$快码源码目录 = Join-Path $快码目录 'src'
$伊里娜目录 = Join-Path $快码源码目录 'irina'
$视窗系统目录 = Join-Path $二进制目录 'OS\windows'
$视窗构建脚本 = Join-Path $快码源码目录 'setup_windows.py'
$合并后的Pyx = Join-Path $快码源码目录 'FasterCode.pyx'
$临时头文件 = Join-Path $快码源码目录 'irina.h'

输出步骤 '检查 Python 解释器'
$Python信息 = 获取Python信息 -Python命令 $Python解释器
$Python版本 = [Version]::new($Python信息.version[0], $Python信息.version[1], $Python信息.version[2])
if ($Python版本 -lt [Version]'3.12.0') {
    throw "需要 Python 3.12 及以上版本，当前为 $Python版本"
}

Write-Host "Python 路径: $($Python信息.executable)"
Write-Host "Python 版本: $Python版本"
Write-Host "用户级 site-packages: $($Python信息.user_site)"

输出步骤 '确保用户级 Python 构建依赖已就绪'
if ($跳过依赖安装) {
    & $Python解释器 -c 'import Cython' *> $null
    if ($LASTEXITCODE -ne 0) {
        throw '缺少 Cython，且已禁止自动安装依赖'
    }
}
else {
    调用并检查 -文件路径 $Python解释器 -参数列表 @('-m', 'pip', 'install', '--user', '--upgrade', 'pip', 'setuptools', 'wheel') -失败提示 '升级 pip、setuptools、wheel 失败'
    确保Python包 -Python命令 $Python解释器 -导入名 'Cython' -包名 'Cython'
    调用并检查 -文件路径 $Python解释器 -参数列表 @('-m', 'pip', 'install', '--user', '-r', (Join-Path $仓库根目录 'requirements.txt')) -失败提示 '安装 Python 运行依赖失败'
}

输出步骤 '查找 MSVC 构建工具'
$VcVars64路径 = 查找VcVars64
if (-not $VcVars64路径) {
    安装构建工具
    $VcVars64路径 = 查找VcVars64
}
if (-not $VcVars64路径) {
    throw '尝试自动安装后仍然无法定位 vcvars64.bat'
}
Write-Host "使用的 vcvars64: $VcVars64路径"

$Pyx源文件列表 = @(
    (Join-Path $快码源码目录 'Faster_Irina.pyx')
    (Join-Path $快码源码目录 'Faster_Polyglot.pyx')
)

$当前Abi标记 = 'cp{0}{1}' -f $Python版本.Major, $Python版本.Minor

输出步骤 '准备 FasterCode 源文件'
$合并后的内容 = (Get-Content $Pyx源文件列表[0] -Raw) + [Environment]::NewLine + (Get-Content $Pyx源文件列表[1] -Raw)
Set-Content -Path $合并后的Pyx -Value $合并后的内容 -Encoding ASCII
$是否复制过头文件 = $false
if (-not (Test-Path $临时头文件)) {
    Copy-Item (Join-Path $伊里娜目录 'irina.h') $临时头文件
    $是否复制过头文件 = $true
}

输出步骤 '按仓库既有 MSVC 流程构建 libirina.lib'
$C源文件列表 = @(
    'lc.c', 'board.c', 'data.c', 'eval.c', 'hash.c', 'loop.c', 'makemove.c',
    'movegen.c', 'movegen_piece_to.c', 'search.c', 'util.c', 'pgn.c', 'parser.c', 'polyglot.c'
)
$目标文件列表 = $C源文件列表 | ForEach-Object { [System.IO.Path]::ChangeExtension($_, '.obj') }
$编译命令 = 'cl /nologo /O2 /DNDEBUG /DWIN32 /MD /c ' + ($C源文件列表 -join ' ')
$制库命令 = 'lib /nologo /OUT:..\irina.lib ' + ($目标文件列表 -join ' ')
$清理命令 = 'del /q ' + ($目标文件列表 -join ' ')
调用Msvc批处理 -VcVars路径 $VcVars64路径 -工作目录 $伊里娜目录 -命令列表 @($编译命令, $制库命令, $清理命令) -失败提示 '构建 libirina.lib 失败'

输出步骤 '调用 setup_windows.py 构建 FasterCode 扩展'
调用Msvc批处理 -VcVars路径 $VcVars64路径 -工作目录 $快码源码目录 -命令列表 @(
    "`"$($Python信息.executable)`" `"$视窗构建脚本`" build_ext --inplace"
) -失败提示 '构建 FasterCode 扩展失败'

输出步骤 '复制构建好的 FasterCode 模块到 bin/OS/windows'
$已构建模块 = Get-ChildItem -Path $快码源码目录 -Filter 'FasterCode*.pyd' | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
if (-not $已构建模块) {
    throw 'FasterCode 构建完成但没有生成 .pyd 文件'
}
if ($已构建模块.Name -notmatch $当前Abi标记) {
    Write-Warning "生成的模块 '$($已构建模块.Name)' 与预期 ABI 标记 '$当前Abi标记' 不一致"
}
Copy-Item $已构建模块.FullName (Join-Path $视窗系统目录 $已构建模块.Name) -Force

输出步骤 '清理 FasterCode 临时构建产物'
Remove-Item (Join-Path $快码源码目录 'build') -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $快码源码目录 'FasterCode.c') -Force -ErrorAction SilentlyContinue
Remove-Item $合并后的Pyx -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $快码源码目录 'irina.lib') -Force -ErrorAction SilentlyContinue
if ($是否复制过头文件) {
    Remove-Item $临时头文件 -Force -ErrorAction SilentlyContinue
}

if (-not $跳过冒烟测试) {
    输出步骤 '执行 FasterCode 导入冒烟测试'
    $冒烟脚本 = @'
import sys
sys.path.insert(0, r"bin")
sys.argv = [r"bin\\LucasR.py", "-healthcheck"]
import Code
import FasterCode
print("FasterCode import OK")
'@
    调用并检查 -文件路径 $Python解释器 -参数列表 @('-c', $冒烟脚本) -工作目录 $仓库根目录 -失败提示 '冒烟测试失败'
}

输出步骤 '构建完成'
Write-Host "已构建模块: $($已构建模块.Name)"
Write-Host "复制目标: $(Join-Path $视窗系统目录 $已构建模块.Name)"