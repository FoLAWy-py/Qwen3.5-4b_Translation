$ErrorActionPreference = 'Stop'
$cleanupRoot = [System.IO.Path]::GetFullPath('D:\Project\witrans').TrimEnd('\')
$planPath = Join-Path $cleanupRoot 'publication\local-cleanup-plan.json'
$receiptPath = Join-Path $cleanupRoot 'publication\local-cleanup-receipt.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Cleanup receipt already exists' }
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if ([System.IO.Path]::GetFullPath($plan.root).TrimEnd('\') -ne $cleanupRoot) { throw 'Wrong plan root' }
$expectedBest = 'fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27'
$bestPath = Join-Path $cleanupRoot 'models\witrans-qwen35-v2-critical-cpo\adapter_model.safetensors'
if ((Get-FileHash -LiteralPath $bestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedBest) { throw 'Best adapter changed' }

function Get-SafeCleanupPath([string]$relative) {
    if ([System.IO.Path]::IsPathRooted($relative)) { throw 'Expected a relative path' }
    $absolute = [System.IO.Path]::GetFullPath((Join-Path $cleanupRoot $relative))
    if (-not $absolute.StartsWith($cleanupRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $relative" }
    $cursor = $absolute
    while ($cursor -ne $cleanupRoot) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $relative" }
        }
        $cursor = Split-Path -Parent $cursor
    }
    return $absolute
}

function Assert-ResultFile([object]$entry) {
    $absolute = Get-SafeCleanupPath $entry.path
    $item = Get-Item -LiteralPath $absolute -Force
    if ($item.PSIsContainer -or $item.Length -ne $entry.bytes) { throw "File type/size changed: $($entry.path)" }
    if ((Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "File changed: $($entry.path)" }
    return [pscustomobject]@{ Path=$absolute; Bytes=[long]$item.Length; WriteTime=$item.LastWriteTimeUtc.Ticks }
}

$activeModules = @('scripts.train_', 'scripts.evaluate_qwen35', 'scripts.run_v', 'scripts.decode_qwen35', 'scripts.benchmark_qwen35', 'scripts.probe_qwen35')
foreach ($process in Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uv.exe'") {
    foreach ($module in $activeModules) {
        if ($process.CommandLine -and $process.CommandLine.Contains($module)) { throw "Active model process $($process.ProcessId)" }
    }
}
$validatedDirectories = @()
foreach ($directory in $plan.remove_model_directories) {
    if (-not $directory.path.StartsWith('models/') -or $directory.path.Split('/').Count -ne 2) { throw 'Unapproved model directory' }
    if ($plan.protected_model_directories -contains $directory.path) { throw 'Protected directory' }
    $absolute = Get-SafeCleanupPath $directory.path
    $children = @(Get-ChildItem -LiteralPath $absolute -Force -Recurse)
    if (@($children | Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReparsePoint }).Count) { throw 'Directory contains a reparse point' }
    if (@($children | Where-Object { -not $_.PSIsContainer }).Count -ne @($directory.files).Count) { throw 'Model directory file set changed' }
    $validated = @()
    foreach ($entry in $directory.files) {
        if (-not $entry.path.StartsWith($directory.path + '/')) { throw 'File is outside declared model directory' }
        $validated += Assert-ResultFile $entry
    }
    $validatedDirectories += [pscustomobject]@{ Path=$absolute; Files=$validated }
}
$extraAllowed = @('v6-cpo-semantic.jsonl','v6-cpo-semantic.summary.json','v6-cpo-unmatched.jsonl','docs/history/README-before-qwen35-20261002.md')
$validatedFiles = @()
foreach ($entry in $plan.remove_files) {
    if (-not $entry.path.StartsWith('runs/') -and $extraAllowed -notcontains $entry.path) { throw 'Unapproved result file' }
    $validatedFiles += Assert-ResultFile $entry
}
Write-Output "Preflight verified: $(@($validatedDirectories).Count) historical model directories and $(@($validatedFiles).Count) result files."
$receipt = [ordered]@{ started_at=[DateTime]::UtcNow.ToString('o'); status='deleting'; user_authorization=$plan.user_authorization; archive_old_results=$false; best_sha256=$expectedBest; removed_model_directories=@(); removed_file_count=0; removed_bytes=0; uploaded=$false }
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
foreach ($directory in $validatedDirectories) {
    foreach ($entry in $directory.Files) {
        $item = Get-Item -LiteralPath $entry.Path -Force
        if ($item.Length -ne $entry.Bytes -or $item.LastWriteTimeUtc.Ticks -ne $entry.WriteTime) { throw 'Model file changed after verification' }
    }
    $absolute = Get-SafeCleanupPath ([System.IO.Path]::GetRelativePath($cleanupRoot,$directory.Path))
    Remove-Item -LiteralPath $absolute -Recurse -Force
    $receipt.removed_model_directories += [System.IO.Path]::GetRelativePath($cleanupRoot,$absolute)
    foreach ($entry in $directory.Files) { $receipt.removed_file_count += 1; $receipt.removed_bytes += $entry.Bytes }
}
foreach ($entry in $validatedFiles) {
    $item = Get-Item -LiteralPath $entry.Path -Force
    if ($item.Length -ne $entry.Bytes -or $item.LastWriteTimeUtc.Ticks -ne $entry.WriteTime) { throw 'Result file changed after verification' }
    $absolute = Get-SafeCleanupPath ([System.IO.Path]::GetRelativePath($cleanupRoot,$entry.Path))
    Remove-Item -LiteralPath $absolute -Force
    $receipt.removed_file_count += 1
    $receipt.removed_bytes += $entry.Bytes
}
foreach ($directory in Get-ChildItem -LiteralPath (Join-Path $cleanupRoot 'runs') -Directory -Recurse -Force | Sort-Object { $_.FullName.Length } -Descending) {
    $absolute = Get-SafeCleanupPath ([System.IO.Path]::GetRelativePath($cleanupRoot,$directory.FullName))
    if (-not @(Get-ChildItem -LiteralPath $absolute -Force).Count) { Remove-Item -LiteralPath $absolute -Force }
}
if ((Get-FileHash -LiteralPath $bestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedBest) { throw 'Best adapter failed final verification' }
if ($receipt.removed_file_count -ne $plan.delete_file_count -or $receipt.removed_bytes -ne $plan.delete_bytes) { throw 'Cleanup count mismatch' }
$receipt.status = 'complete'
$receipt.completed_at = [DateTime]::UtcNow.ToString('o')
$receipt.retained_models = @(Get-ChildItem -LiteralPath (Join-Path $cleanupRoot 'models') -Directory | Select-Object -ExpandProperty Name)
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$receipt | ConvertTo-Json -Depth 5

