
$exe = ".\build-member5\ParallelK_MeansClustering.exe"
$inputFile = ".\benchmark_input.txt"
$processCounts = @(1, 2, 4)
$repetitions = 3

if (!(Test-Path $exe)) {
    throw "Executable not found. Build the project first."
}

if (!(Test-Path $inputFile)) {
    throw "benchmark_input.txt not found."
}

$results = @()

foreach ($p in $processCounts) {
    $times = @()

    for ($run = 1; $run -le $repetitions; $run++) {
        Write-Host "Running P=$p, repetition $run..."

        $output = cmd.exe /c "mpiexec -n $p $exe < $inputFile" 2>&1
        $text = $output -join "`n"

        if ($LASTEXITCODE -ne 0) {
            Write-Host $text
            throw "Benchmark failed for P=$p."
        }

        $match = [regex]::Match(
            $text,
            'Maximum parallel execution time:\s*([0-9.eE+-]+)'
        )

        if (!$match.Success) {
            Write-Host $text
            throw "Could not read execution time for P=$p."
        }

        $elapsed = [double]::Parse(
            $match.Groups[1].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )

        if ($elapsed -le 0) {
            throw "Invalid execution time for P=$p, run $run."
        }

        $times += $elapsed
    }

    $average = ($times | Measure-Object -Average).Average

    if ($average -le 0) {
        throw "Invalid average execution time for P=$p."
    }

    $results += [PSCustomObject]@{
        Processes = $p
        AverageSeconds = $average
    }
}

# Use the one-process execution as the baseline.
$baselineResult = $results |
    Where-Object { $_.Processes -eq 1 }

if ($null -eq $baselineResult -or $baselineResult.AverageSeconds -le 0) {
    throw "Invalid one-process baseline."
}

$baseline = $baselineResult.AverageSeconds

# Calculate speedup and parallel efficiency.
foreach ($r in $results) {
    if ($r.AverageSeconds -le 0 -or $r.Processes -le 0) {
        throw "Invalid performance data for P=$($r.Processes)."
    }

    $speedup = $baseline / $r.AverageSeconds
    $efficiency = 100.0 * $speedup / $r.Processes

    $r | Add-Member -NotePropertyName Speedup `
        -NotePropertyValue $speedup

    $r | Add-Member -NotePropertyName EfficiencyPercent `
        -NotePropertyValue $efficiency
}

# Display results.
Write-Host "`n--- Benchmark Results ---"

$results | Format-Table `
    Processes, `
    @{Name = "Average (s)"; Expression = { "{0:F6}" -f $_.AverageSeconds }}, `
    @{Name = "Speedup"; Expression = { "{0:F3}" -f $_.Speedup }}, `
    @{Name = "Efficiency (%)"; Expression = { "{0:F2}" -f $_.EfficiencyPercent }} `
    -AutoSize

# Export results to CSV.
$results | Export-Csv ".\benchmark_results.csv" -NoTypeInformation

Write-Host "`nSaved results to benchmark_results.csv"
