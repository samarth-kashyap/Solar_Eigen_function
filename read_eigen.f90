program READ_EIGENFUNC

    !defining all the necessary variable necessary

    implicit none
    real*4 arr(100), arr2(100), l, zk, muhz, norm, pi
    real*4, dimension(:), allocatable :: rho, r, dr
    real*4 rsun, msun, rhonorm, freqnorm, speednorm, Gcon
    real*4, dimension(:,:), allocatable :: X, nl_array
    real*4, dimension(:,:), allocatable :: eigU
    real*4, dimension(:,:), allocatable :: eigV
    real*8, dimension(:,:), allocatable :: vars
    integer nr, nl_count, max_nl, norder, norder_max, i, nmax, l_present, max_l

    !the value of pi
    pi = 4.0*DATAN(1.D0)

    !this is what I think is the number of l's are present
    max_nl = 4495
    !max_nl = 10

    max_l = 300 !I know this because I printed the results

    !reading this file first to fetch the value of nr -- the number of radial grids
    open(99, file='egvt.sfopal5h5', form='unformatted', status='old')
    read(99)
    read(99) arr

    nr = floor(arr(21)) - 1

    !nr should be 7305
    print *, nr

    !allocating memories to arrays accordingly
    allocate(rho(nr),r(nr),dr(nr))
    allocate(X(4,nr), eigU(nr,max_nl), eigV(nr,max_nl), vars(6,nr), &
                                                & nl_array(2,max_l))

    !reading file that contains miscellaneous physical quantities
    open(199, file='sfopal5h5', form='unformatted', status='old')
    read(199)
    read(199) arr2

    rsun= arr2(1)   !radius of the Sun
    msun = arr2(2)  !mass of the Sun
    rhonorm = msun/(4.*dble(pi)*rsun*rsun*rsun)  !supposedly this is rho? Factor of 3?

    freqnorm = sqrt(Gcon*msun/(rsun*rsun*rsun))     !mean angular rotation rate
    speednorm = sqrt(Gcon*msun/rsun)    !rotational velocity of surface

    do i=1,nr
        read(199) vars(:,i)
        r(i) = vars(1,i)
        rho(i) = vars(2,i)*vars(6,i)
    enddo
    close(199)
    
    !arranging radius in ascending order
    r = r(nr:1:-1)
    rho = rho(nr:1:-1)

    !computing the radial grid sizes
    dr(1:nr-1) = r(2:nr) - r(1:nr-1)
    dr(nr) = dr(nr-1)

    nl_count = 0
    l = 0
    l_present = 0

    open (unit=1,file='n_and_l.dat',status='replace')
    open (unit=2,file='eigU.dat',status='replace')
    open (unit=3,file='eigV.dat',status='replace')

    !do while (int(l) .le. lmax)
    do while (nl_count < max_nl)
    !do while (l .le. max_l)
        read(99)  norder, l, zk, muhz
        print *, norder, int(l)
        if(int(l) .gt. l_present) then
            print *, norder_max, l_present
            write (1,*) norder_max, l_present
            nl_array(:,l_present + 1) = [norder_max,l_present]
            l_present = l
        endif

        read(99) X
        nl_count = nl_count + 1
        !print *,nl_count
        eigU(:,nl_count) = dble(X(1,nr:1:-1))
        eigV(:,nl_count) = (dble(X(2,nr:1:-1))/rho + dble(X(3,nr:1:-1)))/(r*zk**2)

        !normaliziing the eignevalues. Not sure if this is necessary for us.

        norm = sum(dr*r**2*rho*(eigU(:,nl_count)**2 + l*(l+1)*eigV(:,nl_count)**2))**0.5
        eigU(:,nl_count) = eigU(:,nl_count)/norm
        eigV(:,nl_count) = eigV(:,nl_count)/norm

        !saving the normalized values
        write (2,*) eigU(:,nl_count)
        write (3,*) eigV(:,nl_count)

        norder_max = norder

    enddo

    print *, norder, int(l)
    write (1,*) norder_max, l_present

    close(1)
    close(2)
    close(3)
    close(99)

END program READ_EIGENFUNC