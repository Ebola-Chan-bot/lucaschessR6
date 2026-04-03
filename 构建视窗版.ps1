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
