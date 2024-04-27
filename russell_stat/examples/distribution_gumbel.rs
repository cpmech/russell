use russell_stat::*;

fn main() -> Result<(), StrError> {
    // generate samples
    let mut rng = get_rng();
    let dist = DistributionGumbel::new(0.5, 2.0)?;
    let nsamples = 10_000;
    let mut data = vec![0.0; nsamples];
    for i in 0..nsamples {
        data[i] = dist.sample(&mut rng);
    }
    println!("{}", statistics(&data));

    // text-plot
    let stations = (0..20).map(|i| -5.0 + (i as f64)).collect::<Vec<f64>>();
    let mut hist = Histogram::new(&stations)?;
    hist.set_bar_char('#').set_bar_max_len(40);
    hist.count(&data);
    println!("{}", hist);
    Ok(())
}

/* Sample output

min = -3.8702217016220706
max = 18.48991150178352
mean = 1.68369488450194
std_dev = 2.5805268053167527

[-5,-4) |    0
[-4,-3) |   38
[-3,-2) |  264 #####
[-2,-1) |  929 ###################
[-1, 0) | 1457 ###############################
[ 0, 1) | 1880 ########################################
[ 1, 2) | 1610 ##################################
[ 2, 3) | 1283 ###########################
[ 3, 4) |  910 ###################
[ 4, 5) |  604 ############
[ 5, 6) |  398 ########
[ 6, 7) |  226 ####
[ 7, 8) |  153 ###
[ 8, 9) |   96 ##
[ 9,10) |   57 #
[10,11) |   42
[11,12) |   22
[12,13) |    9
[13,14) |   14
    sum = 9992
*/
