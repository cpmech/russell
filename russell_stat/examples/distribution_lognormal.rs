use russell_stat::*;

fn main() -> Result<(), StrError> {
    // generate samples
    let mut rng = get_rng();
    let dist = DistributionLognormal::new(0.0, 0.25)?;
    let nsamples = 10_000;
    let mut data = vec![0.0; nsamples];
    for i in 0..nsamples {
        data[i] = dist.sample(&mut rng);
    }
    println!("{}", statistics(&data));

    // text-plot
    let stations = (0..25).map(|i| (i as f64) * 0.1).collect::<Vec<f64>>();
    let mut hist = Histogram::new(&stations)?;
    hist.set_bar_char('✨').set_bar_max_len(30);
    hist.count(&data);
    println!("{:.2}", hist);
    Ok(())
}

/* Sample output

min = 0.42738183908592275
max = 2.5343346501352135
mean = 1.0330160154674082
std_dev = 0.2610005570820734

[0.00,0.10) |    0
[0.10,0.20) |    0
[0.20,0.30) |    0
[0.30,0.40) |    0
[0.40,0.50) |   33
[0.50,0.60) |  155 ✨✨
[0.60,0.70) |  558 ✨✨✨✨✨✨✨✨✨✨
[0.70,0.80) | 1092 ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
[0.80,0.90) | 1494 ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
[0.90,1.00) | 1622 ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
[1.00,1.10) | 1509 ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
[1.10,1.20) | 1275 ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
[1.20,1.30) |  817 ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
[1.30,1.40) |  552 ✨✨✨✨✨✨✨✨✨✨
[1.40,1.50) |  364 ✨✨✨✨✨✨
[1.50,1.60) |  202 ✨✨✨
[1.60,1.70) |  151 ✨✨
[1.70,1.80) |   77 ✨
[1.80,1.90) |   49
[1.90,2.00) |   21
[2.00,2.10) |   14
[2.10,2.20) |    7
[2.20,2.30) |    5
[2.30,2.40) |    0
        sum = 9997
*/
