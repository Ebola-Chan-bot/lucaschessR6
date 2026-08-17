param(
    [string]$Python解释器 = '',
    [switch]$跳过依赖安装,
    [switch]$跳过冒烟测试,
    [switch]$跳过启动
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Python路径配置文件 = Join-Path $env:LOCALAPPDATA 'LucasChess\python路径.txt'

function 输出步骤 {
    param([string]$消息)
    Write-Host "`n==> $消息" -ForegroundColor Cyan
}

function 测试Python可用性 {
    param([string]$Python命令)

    if (-not $Python命令) {
        return $false
    }
    try {
        & $Python命令 --version *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function 读取已记住的Python路径 {
    if (-not (Test-Path $Python路径配置文件)) {
        return $null
    }
    $首行内容 = Get-Content -Path $Python路径配置文件 -TotalCount 1
    if (-not $首行内容) {
        return $null
    }
    $保存路径 = $首行内容.Trim().Trim('"').Replace('\\', '\')
    if ((Test-Path $保存路径) -and (测试Python可用性 -Python命令 $保存路径)) {
        return $保存路径
    }
    return $null
}

function 记住Python路径 {
    param([string]$Python路径)

    $规范路径 = [System.IO.Path]::GetFullPath($Python路径.Replace('\\', '\'))
    $配置目录 = Split-Path -Parent $Python路径配置文件
    if (-not (Test-Path $配置目录)) {
        New-Item -ItemType Directory -Path $配置目录 -Force | Out-Null
    }
    Set-Content -Path $Python路径配置文件 -Value $规范路径 -Encoding UTF8
    Write-Host "已记住 Python 解释器路径: $规范路径 ($Python路径配置文件)"
}

function 搜索可用Python {
    foreach ($命令名 in @('python', 'python3', 'py')) {
        $命令信息 = Get-Command -Name $命令名 -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($命令信息 -and (测试Python可用性 -Python命令 $命令信息.Source)) {
            return $命令信息.Source
        }
    }

    $根目录候选列表 = @("$env:LOCALAPPDATA\Programs\Python", $env:ProgramW6432, $env:ProgramFiles, ${env:ProgramFiles(x86)}) | Where-Object { $_ }
    $Python目录列表 = foreach ($根目录 in $根目录候选列表) {
        Get-ChildItem -Path $根目录 -Directory -Filter 'Python*' -ErrorAction SilentlyContinue
    }
    $Python目录列表 = $Python目录列表 | Sort-Object -Property Name -Descending
    foreach ($Python目录 in $Python目录列表) {
        $可执行文件 = Join-Path $Python目录.FullName 'python.exe'
        if ((Test-Path $可执行文件) -and (测试Python可用性 -Python命令 $可执行文件)) {
            return $可执行文件
        }
    }
    return $null
}

function 解析Python解释器 {
    param(
        [string]$指定值,
        [bool]$用户显式指定
    )

    if ($用户显式指定) {
        if (测试Python可用性 -Python命令 $指定值) {
            return $指定值
        }
        throw "指定的 Python 解释器不可用: $指定值"
    }

    $已记住路径 = 读取已记住的Python路径
    if ($已记住路径) {
        Write-Host "使用已记住的 Python 解释器: $已记住路径"
        return $已记住路径
    }

    $搜索到的路径 = 搜索可用Python
    if ($搜索到的路径) {
        return $搜索到的路径
    }

    输出步骤 '未在常见位置找到 Python 解释器'
    Write-Host '请提供 Python 解释器的完整路径（例如 C:\Python312\python.exe），输入后将被记住，下次不再询问。'
    while ($true) {
        $输入路径 = (Read-Host 'Python 解释器路径').Trim().Trim('"')
        if ((Test-Path $输入路径) -and (测试Python可用性 -Python命令 $输入路径)) {
            记住Python路径 -Python路径 $输入路径
            return $输入路径
        }
        Write-Warning "'$输入路径' 无法用作 Python 解释器，请重新输入（Ctrl+C 可取消）。"
    }
}

function 调用并检查 {
    param(
        [string]$文件路径,
        [string[]]$参数列表 = @(),
        [string]$工作目录 = (Get-Location).Path,
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

    $临时脚本文件 = Join-Path $env:TEMP "lucaschess_pyquery_$([guid]::NewGuid().ToString('N')).py"
    Set-Content -Path $临时脚本文件 -Value $查询脚本 -Encoding UTF8
    try {
        $结果文本 = & $Python命令 $临时脚本文件
        if ($LASTEXITCODE -ne 0) {
            throw '无法读取 Python 环境信息'
        }
        return $结果文本 | ConvertFrom-Json
    }
    finally {
        Remove-Item $临时脚本文件 -ErrorAction SilentlyContinue
    }
}

function 测试Python导入 {
    param(
        [string]$Python命令,
        [string]$导入名
    )

    # 导入失败时 python 会把 Traceback 写入 stderr，
    # 在 ErrorActionPreference=Stop 下会直接终止脚本，须用局部 Continue 屏蔽。
    $原有偏好 = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Python命令 -c "import $导入名" *> $null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $原有偏好
    }
}

function 确保Python包 {
    param(
        [string]$Python命令,
        [string]$导入名,
        [string]$包名 = $导入名
    )

    if (测试Python导入 -Python命令 $Python命令 -导入名 $导入名) {
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

function 查看VS安装根目录列表 {
    $安装根目录列表 = [System.Collections.Generic.List[string]]::new()

    $查找器路径 = 查找VsWhere
    if ($查找器路径) {
        $安装路径列表 = @(
            (& $查找器路径 -prerelease -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath),
            (& $查找器路径 -prerelease -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath),
            (& $查找器路径 -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath),
            (& $查找器路径 -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
        ) | Where-Object { $_ }

        foreach ($安装路径 in $安装路径列表) {
            $安装根目录列表.Add($安装路径)
        }
    }

    # 固定扫描常见安装位置（部分新版 VS 可能未被 vswhere 登记，例如 VS 18 Insiders）
    foreach ($VS父目录 in @('C:\Program Files\Microsoft Visual Studio', 'C:\Program Files (x86)\Microsoft Visual Studio')) {
        if (Test-Path $VS父目录) {
            Get-ChildItem -Path $VS父目录 -Directory -Recurse -Depth 1 -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -ne 'Installer' } |
                ForEach-Object { $安装根目录列表.Add($_.FullName) }
        }
    }

    return ($安装根目录列表 | Select-Object -Unique)
}

function 查找完整Msvc工具集版本 {
    param([string]$安装根目录)

    # 同一实例内可能残留多个工具集版本（如半装的旧工具集 include 目录几乎为空），
    # 只有 include 下存在 vcruntime.h 的工具集才是完整的，取版本号最大者。
    $工具集根目录 = Join-Path $安装根目录 'VC\Tools\MSVC'
    if (-not (Test-Path $工具集根目录)) {
        return $null
    }

    $完整版本列表 = Get-ChildItem -Path $工具集根目录 -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'include\vcruntime.h') } |
        Sort-Object -Property @{
            Expression = { if ([Version]::TryParse($_.Name, [ref]$null)) { [Version]$_.Name } else { [Version]'0.0' } }
        } -Descending

    if ($完整版本列表) {
        return $完整版本列表[0].Name
    }
    return $null
}

function 查看VC环境脚本候选列表 {
    # VS 18 起 vcvarsall.bat 已移除，官方入口是 Common7\Tools\VsDevCmd.bat；
    # VS 2022 及更早版本继续使用 VC\Auxiliary\Build\vcvarsall.bat。
    $候选列表 = [System.Collections.Generic.List[object]]::new()
    foreach ($安装根目录 in (查看VS安装根目录列表)) {
        $新版脚本 = Join-Path $安装根目录 'Common7\Tools\VsDevCmd.bat'
        $旧版脚本 = Join-Path $安装根目录 'VC\Auxiliary\Build\vcvarsall.bat'
        if (Test-Path $新版脚本) {
            $新版参数 = [System.Collections.Generic.List[string]]::new()
            $新版参数.Add('-arch=amd64')
            $新版参数.Add('-no_logo')
            # 避免默认选中残留的残缺工具集，显式锁定最新的完整工具集
            $完整工具集版本 = 查找完整Msvc工具集版本 -安装根目录 $安装根目录
            if ($完整工具集版本) {
                $新版参数.Add("-vcvars_ver=$完整工具集版本")
            }
            $候选列表.Add(@{ 脚本 = $新版脚本; 参数 = $新版参数.ToArray() })
        }
        elseif (Test-Path $旧版脚本) {
            $候选列表.Add(@{ 脚本 = $旧版脚本; 参数 = @('x64') })
        }
    }
    return $候选列表
}

function 查找VC环境脚本 {
    foreach ($候选 in (查看VC环境脚本候选列表)) {
        Write-Host "验证 MSVC 工具链: $($候选.脚本)"
        if (测试Msvc工具链可用 -环境脚本 $候选.脚本 -环境参数 $候选.参数) {
            return $候选
        }
        Write-Warning "'$($候选.脚本)' 存在但 MSVC 编译器不可用，跳过"
    }

    return $null
}

function 测试Msvc工具链可用 {
    param(
        [string]$环境脚本,
        [string[]]$环境参数 = @()
    )

    # 必须以真实编译校验：个别残留脚本（如已废弃的 vcvars64.bat）虽然存在但无法初始化环境；
    # 测试文件须引用系统头文件，否则残缺工具集（缺 vcruntime.h 等）会漏检。
    $测试目录 = Join-Path $env:TEMP ("lucaschess_cl_test_{0}" -f ([guid]::NewGuid().ToString('N')))
    New-Item -ItemType Directory -Path $测试目录 -Force | Out-Null
    try {
        Set-Content -Path (Join-Path $测试目录 'test.c') -Value "#include <stdio.h>`r`nint main(void){return 0;}" -Encoding ASCII
        $参数文本 = if ($环境参数.Count -gt 0) { ' ' + ($环境参数 -join ' ') } else { '' }
        cmd.exe /c "call `"$环境脚本`"$参数文本 >nul 2>&1 && cd /d `"$测试目录`" && cl /nologo /c test.c >nul 2>&1" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        Remove-Item $测试目录 -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function 调用Msvc批处理 {
    param(
        [string]$环境脚本,
        [string[]]$环境参数,
        [string]$工作目录,
        [string[]]$命令列表,
        [string]$失败提示
    )

    $参数文本 = if ($环境参数.Count -gt 0) { ' ' + ($环境参数 -join ' ') } else { '' }
    $临时脚本 = Join-Path $env:TEMP ("lucaschess_build_{0}.cmd" -f ([guid]::NewGuid().ToString('N')))
    $脚本内容 = @(
        '@echo off',
        'setlocal',
        "call `"$环境脚本`"$参数文本",
        'if errorlevel 1 exit /b 1',
        "cd /d `"$工作目录`""
    ) + $命令列表
    # 必须按系统 ANSI 编码写入（默认 GBK）：脚本内含带中文用户名的 Python 路径，
    # 若写为 ASCII 会导致中文损坏，cmd 报"系统找不到指定的路径"
    Set-Content -Path $临时脚本 -Value $脚本内容 -Encoding Default

    try {
        # 注意：不能用 cmd /d，它会让 vcvars64.bat 内部对带空格路径的 call 解析失败
        调用并检查 -文件路径 'cmd.exe' -参数列表 @('/c', $临时脚本) -失败提示 $失败提示
    }
    finally {
        Remove-Item $临时脚本 -ErrorAction SilentlyContinue
    }
}

function 查找仓库根目录 {
    param([string]$起始目录)

    function 是有效仓库 ([string]$目录) {
        return ((Test-Path (Join-Path $目录 'requirements.txt')) -and
                (Test-Path (Join-Path $目录 'bin\LucasR.py')))
    }

    # 脚本自身所在目录（上游布局：脚本在仓库根）
    if (是有效仓库 $起始目录) {
        return $起始目录
    }

    # 脚本位于部署工具等子目录时，探测同一工作区下的兄弟仓库目录
    $父目录 = Split-Path -Parent $起始目录
    foreach ($子项 in (Get-ChildItem -Path $父目录 -Directory -ErrorAction SilentlyContinue)) {
        if (是有效仓库 $子项.FullName) {
            return $子项.FullName
        }
    }

    throw "无法定位 LucasChess 仓库根目录（需包含 requirements.txt 与 bin\LucasR.py），已探测: $起始目录 及其子目录"
}

$仓库根目录 = 查找仓库根目录 -起始目录 (Split-Path -Parent $MyInvocation.MyCommand.Path)
Write-Host "仓库根目录: $仓库根目录"
$二进制目录 = Join-Path $仓库根目录 'bin'
$快码目录 = Join-Path $二进制目录 '_fastercode'
$快码源码目录 = Join-Path $快码目录 'src'
$伊里娜目录 = Join-Path $快码源码目录 'irina'
# 合并 main(R6.0.4) 后，bin/Code/__init__.py 使用 platform = "windows"，
# 即应用从 bin/OS/windows 目录加载 FasterCode 扩展（与上游构建 bat 一致）
$视窗系统目录 = Join-Path $二进制目录 'OS\windows'
$视窗构建脚本 = Join-Path $快码源码目录 'setup_windows.py'
$合并后的Pyx = Join-Path $快码源码目录 'FasterCode.pyx'
$临时头文件 = Join-Path $快码源码目录 'irina.h'

输出步骤 '检查 Python 解释器'
$Python解释器 = 解析Python解释器 -指定值 $Python解释器 -用户显式指定 ($PSBoundParameters.ContainsKey('Python解释器'))
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
    if (-not (测试Python导入 -Python命令 $Python解释器 -导入名 'Cython')) {
        throw '缺少 Cython，且已禁止自动安装依赖'
    }
}
else {
    调用并检查 -文件路径 $Python解释器 -参数列表 @('-m', 'pip', 'install', '--user', '--upgrade', 'pip', 'setuptools', 'wheel') -失败提示 '升级 pip、setuptools、wheel 失败'
    确保Python包 -Python命令 $Python解释器 -导入名 'Cython' -包名 'Cython'
    调用并检查 -文件路径 $Python解释器 -参数列表 @('-m', 'pip', 'install', '--user', '-r', (Join-Path $仓库根目录 'requirements.txt')) -失败提示 '安装 Python 运行依赖失败'
}

输出步骤 '查找 MSVC 构建工具'
$VC环境 = 查找VC环境脚本
if (-not $VC环境) {
    throw '未找到可用的 MSVC 构建工具链（VsDevCmd.bat / vcvarsall.bat 均不可用）。请在 Visual Studio Installer 中为已有的 VS 实例勾选"使用 C++ 的桌面开发"工作负载'
}
Write-Host "使用的 VC 构建环境脚本: $($VC环境.脚本)"

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
调用Msvc批处理 -环境脚本 $VC环境.脚本 -环境参数 $VC环境.参数 -工作目录 $伊里娜目录 -命令列表 @($编译命令, $制库命令, $清理命令) -失败提示 '构建 libirina.lib 失败'

输出步骤 '调用 setup_windows.py 构建 FasterCode 扩展'
调用Msvc批处理 -环境脚本 $VC环境.脚本 -环境参数 $VC环境.参数 -工作目录 $快码源码目录 -命令列表 @(
    'set DISTUTILS_USE_SDK=1',
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
if (-not (Test-Path $视窗系统目录)) {
    New-Item -ItemType Directory -Path $视窗系统目录 -Force | Out-Null
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
    输出步骤 '执行冒烟测试（导入完整启动链路）'
    # 必须导入 Code.Main.Init：它会级联加载 Procesador 及全部模块，
    # 仅 import Code 只触发包级 __init__，无法发现深层模块的接口错配（如缺常量等）。
    $冒烟脚本 = @'
import sys
sys.path.insert(0, r"bin")
sys.argv = [r"bin\\LucasR.py", "-healthcheck"]

import Code
import FasterCode
import Code.Main.Init
print("冒烟测试 OK：Code.Main.Init / FasterCode 导入成功")
'@
    $临时冒烟文件 = Join-Path $env:TEMP "lucaschess_smoke_$([guid]::NewGuid().ToString('N')).py"
    Set-Content -Path $临时冒烟文件 -Value $冒烟脚本 -Encoding UTF8
    try {
        调用并检查 -文件路径 $Python解释器 -参数列表 @($临时冒烟文件) -工作目录 $仓库根目录 -失败提示 '冒烟测试失败'
    }
    finally {
        Remove-Item $临时冒烟文件 -ErrorAction SilentlyContinue
    }
}

输出步骤 '构建完成'
Write-Host "已构建模块: $($已构建模块.Name)"
Write-Host "复制目标: $(Join-Path $视窗系统目录 $已构建模块.Name)"

if (-not $跳过启动) {
    输出步骤 '启动 LucasChess'
    $启动入口 = Join-Path $二进制目录 'LucasR.py'
    if (-not (Test-Path $启动入口)) {
        throw "启动入口不存在: $启动入口"
    }
    Start-Process -FilePath $Python解释器 -ArgumentList @($启动入口) -WorkingDirectory $二进制目录
    Write-Host 'LucasChess 已在后台启动'
}
# SIG # Begin signature block
# MIIbygYJKoZIhvcNAQcCoIIbuzCCG7cCAQExCzAJBgUrDgMCGgUAMGkGCisGAQQB
# gjcCAQSgWzBZMDQGCisGAQQBgjcCAR4wJgIDAQAABBAfzDtgWUsITrck0sYpfvNR
# AgEAAgEAAgEAAgEAAgEAMCEwCQYFKw4DAhoFAAQUSAhIKI9gKGsmFr6Og92o33JR
# rUegghY+MIIDADCCAeigAwIBAgIQGEkdcRrke6ZOB0aPBLgcCjANBgkqhkiG9w0B
# AQsFADAXMRUwEwYDVQQDDAzln4PljZrmi4nphbEwIBcNMjUwMTEzMDkzNjU2WhgP
# MjEyNTAxMTMwOTQ2NTZaMBcxFTATBgNVBAMMDOWfg+WNmuaLiemFsTCCASIwDQYJ
# KoZIhvcNAQEBBQADggEPADCCAQoCggEBAMRKufgmQjHzHMxJNEtYKQLHGh7zfjzN
# hWU8GuFZOV9XQUtBOFM6+Mf8SBoHHId0SNQP4uvxIRarsAK6fSf9MAOxUjoH8ppB
# 6T++Plrakl/5nEgNVntaojTcMyuuiWfEE8mMHoHsZOtbw4rz69P4DHILFLXsS5L5
# Du4lxMXqYkbcyBzuz8SySqlE73MnYhRPq5m75uqx/IGpx4RPDwS25HkrypZwwAeo
# dhHDRmo3xs8cH+x59lmRNtRdnJxlzfIviZSkdy6zYhN0zmaeazom4gj9VcOoM96i
# kJcdbgFYD7GDRyByD2yGM8HyqA4NEuVKiTQ2aMIRpAyxfOLxU+VxWrECAwEAAaNG
# MEQwDgYDVR0PAQH/BAQDAgeAMBMGA1UdJQQMMAoGCCsGAQUFBwMDMB0GA1UdDgQW
# BBRQz+QeZSmdXRWo6j1TKt879B7K1DANBgkqhkiG9w0BAQsFAAOCAQEAFVTMwI1V
# 8C30j/c/RbhUt/1jjpB79CvgRFmjfYZEumZ1FNxc2HVJ8Lf4jqo2VsMiWP2/VQTu
# 4ZvWWajBGjA5sx+DTcqR1U/iU5WAc/vHyYw+osbF9uN1g2p2qtsr8TOcbwFdmRLw
# cuaqLX9sxY3WT0l2fKVxzw00ljyXMGePlq1KRBYgy7X+h7DJ51R6d5satDcLL6oS
# LVi9spFVo7tNjQ4xL1ImgF1XzZe767Ye/RmUMNO29db052KJlKpymHeKg/AKJ2BW
# XeGB0ISeVFPC5fm7ya6whQlpdeLLvpKyunez1D6qTdO4ZUFdSSH8Q9JIkx6TiTEy
# BqlK+Hq5jZtuMDCCBY0wggR1oAMCAQICEA6bGI750C3n79tQ4ghAGFowDQYJKoZI
# hvcNAQEMBQAwZTELMAkGA1UEBhMCVVMxFTATBgNVBAoTDERpZ2lDZXJ0IEluYzEZ
# MBcGA1UECxMQd3d3LmRpZ2ljZXJ0LmNvbTEkMCIGA1UEAxMbRGlnaUNlcnQgQXNz
# dXJlZCBJRCBSb290IENBMB4XDTIyMDgwMTAwMDAwMFoXDTMxMTEwOTIzNTk1OVow
# YjELMAkGA1UEBhMCVVMxFTATBgNVBAoTDERpZ2lDZXJ0IEluYzEZMBcGA1UECxMQ
# d3d3LmRpZ2ljZXJ0LmNvbTEhMB8GA1UEAxMYRGlnaUNlcnQgVHJ1c3RlZCBSb290
# IEc0MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAv+aQc2jeu+RdSjww
# IjBpM+zCpyUuySE98orYWcLhKac9WKt2ms2uexuEDcQwH/MbpDgW61bGl20dq7J5
# 8soR0uRf1gU8Ug9SH8aeFaV+vp+pVxZZVXKvaJNwwrK6dZlqczKU0RBEEC7fgvMH
# hOZ0O21x4i0MG+4g1ckgHWMpLc7sXk7Ik/ghYZs06wXGXuxbGrzryc/NrDRAX7F6
# Zu53yEioZldXn1RYjgwrt0+nMNlW7sp7XeOtyU9e5TXnMcvak17cjo+A2raRmECQ
# ecN4x7axxLVqGDgDEI3Y1DekLgV9iPWCPhCRcKtVgkEy19sEcypukQF8IUzUvK4b
# A3VdeGbZOjFEmjNAvwjXWkmkwuapoGfdpCe8oU85tRFYF/ckXEaPZPfBaYh2mHY9
# WV1CdoeJl2l6SPDgohIbZpp0yt5LHucOY67m1O+SkjqePdwA5EUlibaaRBkrfsCU
# tNJhbesz2cXfSwQAzH0clcOP9yGyshG3u3/y1YxwLEFgqrFjGESVGnZifvaAsPvo
# ZKYz0YkH4b235kOkGLimdwHhD5QMIR2yVCkliWzlDlJRR3S+Jqy2QXXeeqxfjT/J
# vNNBERJb5RBQ6zHFynIWIgnffEx1P2PsIV/EIFFrb7GrhotPwtZFX50g/KEexcCP
# orF+CiaZ9eRpL5gdLfXZqbId5RsCAwEAAaOCATowggE2MA8GA1UdEwEB/wQFMAMB
# Af8wHQYDVR0OBBYEFOzX44LScV1kTN8uZz/nupiuHA9PMB8GA1UdIwQYMBaAFEXr
# oq/0ksuCMS1Ri6enIZ3zbcgPMA4GA1UdDwEB/wQEAwIBhjB5BggrBgEFBQcBAQRt
# MGswJAYIKwYBBQUHMAGGGGh0dHA6Ly9vY3NwLmRpZ2ljZXJ0LmNvbTBDBggrBgEF
# BQcwAoY3aHR0cDovL2NhY2VydHMuZGlnaWNlcnQuY29tL0RpZ2lDZXJ0QXNzdXJl
# ZElEUm9vdENBLmNydDBFBgNVHR8EPjA8MDqgOKA2hjRodHRwOi8vY3JsMy5kaWdp
# Y2VydC5jb20vRGlnaUNlcnRBc3N1cmVkSURSb290Q0EuY3JsMBEGA1UdIAQKMAgw
# BgYEVR0gADANBgkqhkiG9w0BAQwFAAOCAQEAcKC/Q1xV5zhfoKN0Gz22Ftf3v1cH
# vZqsoYcs7IVeqRq7IviHGmlUIu2kiHdtvRoU9BNKei8ttzjv9P+Aufih9/Jy3iS8
# UgPITtAq3votVs/59PesMHqai7Je1M/RQ0SbQyHrlnKhSLSZy51PpwYDE3cnRNTn
# f+hZqPC/Lwum6fI0POz3A8eHqNJMQBk1RmppVLC4oVaO7KTVPeix3P0c2PR3WlxU
# jG/voVA9/HYJaISfb8rbII01YBwCA8sgsKxYoA5AY8WYIsGyWfVVa88nq2x2zm8j
# LfR+cWojayL/ErhULSd+2DrZ8LaHlv1b0VysGMNNn3O3AamfV6peKOK5lDCCBrQw
# ggScoAMCAQICEA3HrFcF/yGZLkBDIgw6SYYwDQYJKoZIhvcNAQELBQAwYjELMAkG
# A1UEBhMCVVMxFTATBgNVBAoTDERpZ2lDZXJ0IEluYzEZMBcGA1UECxMQd3d3LmRp
# Z2ljZXJ0LmNvbTEhMB8GA1UEAxMYRGlnaUNlcnQgVHJ1c3RlZCBSb290IEc0MB4X
# DTI1MDUwNzAwMDAwMFoXDTM4MDExNDIzNTk1OVowaTELMAkGA1UEBhMCVVMxFzAV
# BgNVBAoTDkRpZ2lDZXJ0LCBJbmMuMUEwPwYDVQQDEzhEaWdpQ2VydCBUcnVzdGVk
# IEc0IFRpbWVTdGFtcGluZyBSU0E0MDk2IFNIQTI1NiAyMDI1IENBMTCCAiIwDQYJ
# KoZIhvcNAQEBBQADggIPADCCAgoCggIBALR4MdMKmEFyvjxGwBysddujRmh0tFEX
# nU2tjQ2UtZmWgyxU7UNqEY81FzJsQqr5G7A6c+Gh/qm8Xi4aPCOo2N8S9SLrC6Kb
# ltqn7SWCWgzbNfiR+2fkHUiljNOqnIVD/gG3SYDEAd4dg2dDGpeZGKe+42DFUF0m
# R/vtLa4+gKPsYfwEu7EEbkC9+0F2w4QJLVSTEG8yAR2CQWIM1iI5PHg62IVwxKSp
# O0XaF9DPfNBKS7Zazch8NF5vp7eaZ2CVNxpqumzTCNSOxm+SAWSuIr21Qomb+zzQ
# WKhxKTVVgtmUPAW35xUUFREmDrMxSNlr/NsJyUXzdtFUUt4aS4CEeIY8y9IaaGBp
# PNXKFifinT7zL2gdFpBP9qh8SdLnEut/GcalNeJQ55IuwnKCgs+nrpuQNfVmUB5K
# lCX3ZA4x5HHKS+rqBvKWxdCyQEEGcbLe1b8Aw4wJkhU1JrPsFfxW1gaou30yZ46t
# 4Y9F20HHfIY4/6vHespYMQmUiote8ladjS/nJ0+k6MvqzfpzPDOy5y6gqztiT96F
# v/9bH7mQyogxG9QEPHrPV6/7umw052AkyiLA6tQbZl1KhBtTasySkuJDpsZGKdls
# jg4u70EwgWbVRSX1Wd4+zoFpp4Ra+MlKM2baoD6x0VR4RjSpWM8o5a6D8bpfm4CL
# KczsG7ZrIGNTAgMBAAGjggFdMIIBWTASBgNVHRMBAf8ECDAGAQH/AgEAMB0GA1Ud
# DgQWBBTvb1NK6eQGfHrK4pBW9i/USezLTjAfBgNVHSMEGDAWgBTs1+OC0nFdZEzf
# Lmc/57qYrhwPTzAOBgNVHQ8BAf8EBAMCAYYwEwYDVR0lBAwwCgYIKwYBBQUHAwgw
# dwYIKwYBBQUHAQEEazBpMCQGCCsGAQUFBzABhhhodHRwOi8vb2NzcC5kaWdpY2Vy
# dC5jb20wQQYIKwYBBQUHMAKGNWh0dHA6Ly9jYWNlcnRzLmRpZ2ljZXJ0LmNvbS9E
# aWdpQ2VydFRydXN0ZWRSb290RzQuY3J0MEMGA1UdHwQ8MDowOKA2oDSGMmh0dHA6
# Ly9jcmwzLmRpZ2ljZXJ0LmNvbS9EaWdpQ2VydFRydXN0ZWRSb290RzQuY3JsMCAG
# A1UdIAQZMBcwCAYGZ4EMAQQCMAsGCWCGSAGG/WwHATANBgkqhkiG9w0BAQsFAAOC
# AgEAF877FoAc/gc9EXZxML2+C8i1NKZ/zdCHxYgaMH9Pw5tcBnPw6O6FTGNpoV2V
# 4wzSUGvI9NAzaoQk97frPBtIj+ZLzdp+yXdhOP4hCFATuNT+ReOPK0mCefSG+tXq
# GpYZ3essBS3q8nL2UwM+NMvEuBd/2vmdYxDCvwzJv2sRUoKEfJ+nN57mQfQXwcAE
# GCvRR2qKtntujB71WPYAgwPyWLKu6RnaID/B0ba2H3LUiwDRAXx1Neq9ydOal95C
# HfmTnM4I+ZI2rVQfjXQA1WSjjf4J2a7jLzWGNqNX+DF0SQzHU0pTi4dBwp9nEC8E
# AqoxW6q17r0z0noDjs6+BFo+z7bKSBwZXTRNivYuve3L2oiKNqetRHdqfMTCW/Nm
# KLJ9M+MtucVGyOxiDf06VXxyKkOirv6o02OoXN4bFzK0vlNMsvhlqgF2puE6Fndl
# ENSmE+9JGYxOGLS/D284NHNboDGcmWXfwXRy4kbu4QFhOm0xJuF2EZAOk5eCkhSx
# ZON3rGlHqhpB/8MluDezooIs8CVnrpHMiD2wL40mm53+/j7tFaxYKIqL0Q4ssd8x
# HZnIn/7GELH3IdvG2XlM9q7WP/UwgOkw/HQtyRN62JK4S1C8uw3PdBunvAZapsiI
# 5YKdvlarEvf8EA+8hcpSM9LHJmyrxaFtoza2zNaQ9k+5t1wwggbtMIIE1aADAgEC
# AhAKgO8YS43xBYLRxHanlXRoMA0GCSqGSIb3DQEBCwUAMGkxCzAJBgNVBAYTAlVT
# MRcwFQYDVQQKEw5EaWdpQ2VydCwgSW5jLjFBMD8GA1UEAxM4RGlnaUNlcnQgVHJ1
# c3RlZCBHNCBUaW1lU3RhbXBpbmcgUlNBNDA5NiBTSEEyNTYgMjAyNSBDQTEwHhcN
# MjUwNjA0MDAwMDAwWhcNMzYwOTAzMjM1OTU5WjBjMQswCQYDVQQGEwJVUzEXMBUG
# A1UEChMORGlnaUNlcnQsIEluYy4xOzA5BgNVBAMTMkRpZ2lDZXJ0IFNIQTI1NiBS
# U0E0MDk2IFRpbWVzdGFtcCBSZXNwb25kZXIgMjAyNSAxMIICIjANBgkqhkiG9w0B
# AQEFAAOCAg8AMIICCgKCAgEA0EasLRLGntDqrmBWsytXum9R/4ZwCgHfyjfMGUIw
# YzKomd8U1nH7C8Dr0cVMF3BsfAFI54um8+dnxk36+jx0Tb+k+87H9WPxNyFPJIDZ
# HhAqlUPt281mHrBbZHqRK71Em3/hCGC5KyyneqiZ7syvFXJ9A72wzHpkBaMUNg7M
# OLxI6E9RaUueHTQKWXymOtRwJXcrcTTPPT2V1D/+cFllESviH8YjoPFvZSjKs3SK
# O1QNUdFd2adw44wDcKgH+JRJE5Qg0NP3yiSyi5MxgU6cehGHr7zou1znOM8odbkq
# oK+lJ25LCHBSai25CFyD23DZgPfDrJJJK77epTwMP6eKA0kWa3osAe8fcpK40uhk
# tzUd/Yk0xUvhDU6lvJukx7jphx40DQt82yepyekl4i0r8OEps/FNO4ahfvAk12hE
# 5FVs9HVVWcO5J4dVmVzix4A77p3awLbr89A90/nWGjXMGn7FQhmSlIUDy9Z2hSgc
# taepZTd0ILIUbWuhKuAeNIeWrzHKYueMJtItnj2Q+aTyLLKLM0MheP/9w6CtjuuV
# HJOVoIJ/DtpJRE7Ce7vMRHoRon4CWIvuiNN1Lk9Y+xZ66lazs2kKFSTnnkrT3pXW
# ETTJkhd76CIDBbTRofOsNyEhzZtCGmnQigpFHti58CSmvEyJcAlDVcKacJ+A9/z7
# eacCAwEAAaOCAZUwggGRMAwGA1UdEwEB/wQCMAAwHQYDVR0OBBYEFOQ7/PIx7f39
# 1/ORcWMZUEPPYYzoMB8GA1UdIwQYMBaAFO9vU0rp5AZ8esrikFb2L9RJ7MtOMA4G
# A1UdDwEB/wQEAwIHgDAWBgNVHSUBAf8EDDAKBggrBgEFBQcDCDCBlQYIKwYBBQUH
# AQEEgYgwgYUwJAYIKwYBBQUHMAGGGGh0dHA6Ly9vY3NwLmRpZ2ljZXJ0LmNvbTBd
# BggrBgEFBQcwAoZRaHR0cDovL2NhY2VydHMuZGlnaWNlcnQuY29tL0RpZ2lDZXJ0
# VHJ1c3RlZEc0VGltZVN0YW1waW5nUlNBNDA5NlNIQTI1NjIwMjVDQTEuY3J0MF8G
# A1UdHwRYMFYwVKBSoFCGTmh0dHA6Ly9jcmwzLmRpZ2ljZXJ0LmNvbS9EaWdpQ2Vy
# dFRydXN0ZWRHNFRpbWVTdGFtcGluZ1JTQTQwOTZTSEEyNTYyMDI1Q0ExLmNybDAg
# BgNVHSAEGTAXMAgGBmeBDAEEAjALBglghkgBhv1sBwEwDQYJKoZIhvcNAQELBQAD
# ggIBAGUqrfEcJwS5rmBB7NEIRJ5jQHIh+OT2Ik/bNYulCrVvhREafBYF0RkP2AGr
# 181o2YWPoSHz9iZEN/FPsLSTwVQWo2H62yGBvg7ouCODwrx6ULj6hYKqdT8wv2UV
# +Kbz/3ImZlJ7YXwBD9R0oU62PtgxOao872bOySCILdBghQ/ZLcdC8cbUUO75ZSpb
# h1oipOhcUT8lD8QAGB9lctZTTOJM3pHfKBAEcxQFoHlt2s9sXoxFizTeHihsQyfF
# g5fxUFEp7W42fNBVN4ueLaceRf9Cq9ec1v5iQMWTFQa0xNqItH3CPFTG7aEQJmmr
# JTV3Qhtfparz+BW60OiMEgV5GWoBy4RVPRwqxv7Mk0Sy4QHs7v9y69NBqycz0BZw
# hB9WOfOu/CIJnzkQTwtSSpGGhLdjnQ4eBpjtP+XB3pQCtv4E5UCSDag6+iX8MmB1
# 0nfldPF9SVD7weCC3yXZi/uuhqdwkgVxuiMFzGVFwYbQsiGnoa9F5AaAyBjFBtXV
# LcKtapnMG3VH3EmAp/jsJ3FVF3+d1SVDTmjFjLbNFZUWMXuZyvgLfgyPehwJVxwC
# +UpX2MSey2ueIu9THFVkT+um1vshETaWyQo8gmBto/m3acaP9QsuLj3FNwFlTxq2
# 5+T4QwX9xa6ILs84ZPvmpovq90K8eWyG2N01c4IhSOxqt81nMYIE9jCCBPICAQEw
# KzAXMRUwEwYDVQQDDAzln4PljZrmi4nphbECEBhJHXEa5HumTgdGjwS4HAowCQYF
# Kw4DAhoFAKB4MBgGCisGAQQBgjcCAQwxCjAIoAKAAKECgAAwGQYJKoZIhvcNAQkD
# MQwGCisGAQQBgjcCAQQwHAYKKwYBBAGCNwIBCzEOMAwGCisGAQQBgjcCARUwIwYJ
# KoZIhvcNAQkEMRYEFKhP38EPhehkA13ez6nmozkoWU3mMA0GCSqGSIb3DQEBAQUA
# BIIBAC4atNokN4qaw0u68XqfP0Wf2EGZMoA10tGOfq+lhaN4RzaUxts5PFk1M9hU
# n3H5gqXWXqPebx5foWACZVVQmAYsLfpbZWaFHEUWiz6t/L8ce2ust73WAV+v/aoN
# 6B07xIidFHvn0bXK+d4vtIKrBQxQdV32sX58wrXvdyz4Rc7j4S3cY0m0RIqt1xi6
# O/GOKSrglB71NR3CYANi+q8W6nQgxrFVeL16N/23Pxdrx9Ru+iCiD9WExLJYoinj
# F8Dmn+TQlmeMu96Ok97Y4HHK9xUE5bx1jpKpRpAyDe22nCsX/MTEkB6Z7nZx9oaW
# fhCqUTYhqJUjcdnvB3lOCxXCz3yhggMmMIIDIgYJKoZIhvcNAQkGMYIDEzCCAw8C
# AQEwfTBpMQswCQYDVQQGEwJVUzEXMBUGA1UEChMORGlnaUNlcnQsIEluYy4xQTA/
# BgNVBAMTOERpZ2lDZXJ0IFRydXN0ZWQgRzQgVGltZVN0YW1waW5nIFJTQTQwOTYg
# U0hBMjU2IDIwMjUgQ0ExAhAKgO8YS43xBYLRxHanlXRoMA0GCWCGSAFlAwQCAQUA
# oGkwGAYJKoZIhvcNAQkDMQsGCSqGSIb3DQEHATAcBgkqhkiG9w0BCQUxDxcNMjYw
# NDAzMDUzMTA5WjAvBgkqhkiG9w0BCQQxIgQg8INE0bvRCT4A9C2Dgq9shK7YW6KG
# +ncHKd12pMu1xaswDQYJKoZIhvcNAQEBBQAEggIAdchDBQBwjqEef5CfPTHa68f+
# 3U2y+d+/cddrhHAMs4Dm/rBny1Iryef0QNZ0GPgkNhOY/Vfwt0UpPzu/lFKAHVm+
# it38tLO97KDTxhSkco+LFf5g1UCcv2yav87B4fE+aVl7SEmFmQFsZFhK9wxshcT0
# Ls38guKDfgYdC1xFb9bxk1tygl33kLAtmedipyYf8iGUzCRuP5+GKe0wVZOhru/O
# kzgLxhMu4RreWy23KttMwrGVrc31JAf58ws/sJyFSCwi0tjh2OhO+5QuqItlIvpz
# Y9BOEyjNuNU4ygNmzIRMLNREjQ9e/Vy+LE96g2UuBoaLXezx2FTHrwUmV9rT2JKj
# jwDWYgsSDPLBw+9oy8nlheqrqSIW7LC3BV9BIgWlNEqpP6VR6tcA4SlCV7dWt+gW
# HuhurQuOgmiNmj1rTepOtvUgShI5/Wz9T3dxWjKCVFDwcnAyjR88AD1SzOBLta0N
# s5wEJofI1mpcqaJTlJHNcDrUJTqpn++TSE85RHz3GsZp286uXuSrLpqholoH6hFv
# pwq8aUBZJGlpOyO1rg8854rlXh3MQOOEOVQc7SA4YGNFilv/UgTzKrusBOf4cKJu
# L4r8aFh2uVgqnRNrZWm/avRBT+OvSPFFdi+ORuthss991dqunrcYqZjuicwC6KRq
# cmSBRgV/jRDxfPneKk8=
# SIG # End signature block