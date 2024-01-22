use russell_lab::Vector;

// Holds workspace variables for the Runge-Kutta method
pub struct Workspace {
    // workspace
    pub nstg: usize,    // number of stages
    pub ndim: usize,    // dimension of y vector
    pub u: Vector,      // u[stg] = x + h*c[stg]
    pub v: Vec<Vector>, // v[stg][dim] = ya[dim] + h*sum(a[stg][j]*f[j][dim], j, nstg)
    pub f: Vec<Vector>, // f[stg][dim] = f(u[stg], v[stg][dim])

    // step data
    pub rs: f64,      // stiffness ratio ρ = ‖ k[s] - k[s-1] ‖ / ‖ v[s] - v[s-1] ‖
    pub h: f64,       // current stepsize
    pub h_prev: f64,  // previous stepsize
    pub first: bool,  // first step
    pub f0: Vector,   // f(x,y) before step
    pub scal: Vector, // scal = Atol + Rtol*abs(y)

    // step control data
    pub reuse_jac_and_dec_once: bool, // reuse current Jacobian and current decomposition
    pub reuse_jac_once: bool,         // reuse last Jacobian (only)
    pub jac_is_ok: bool,              // Jacobian is OK
    pub nit: usize,                   // current number of iterations
    pub eta: f64,                     // eta tolerance
    pub theta: f64,                   // theta variable
    pub dvfac: f64,                   // divergence factor
    pub diverg: bool,                 // flag diverging step
    pub reject: bool,                 // reject step

    // error control
    pub rerr: f64,      // relative error
    pub rerr_prev: f64, // previous relative error

    // stiffness detection
    pub stiff_yes: usize, // counter of "stiff" steps
    pub stiff_not: usize, // counter of not "stiff" steps
}

impl Workspace {
    // new_rk_work returns a new structure
    pub fn new(nstg: usize, ndim: usize) -> Self {
        // workspace
        let mut u = Vector::new(nstg);
        let mut v = vec![Vector::new(ndim); nstg];
        let mut f = vec![Vector::new(ndim); nstg];

        // step data
        let f0 = Vector::new(ndim);
        let scal = Vector::new(ndim);

        Workspace {
            nstg,
            ndim,
            u,
            v,
            f,
            rs: 0.0,
            h: 0.0,
            h_prev: 0.0,
            first: false,
            f0,
            scal,
            reuse_jac_and_dec_once: false,
            reuse_jac_once: false,
            jac_is_ok: false,
            nit: 0,
            eta: 0.0,
            theta: 0.0,
            dvfac: 0.0,
            diverg: false,
            reject: false,
            rerr: 0.0,
            rerr_prev: 0.0,
            stiff_yes: 0,
            stiff_not: 0,
        }
    }
}
