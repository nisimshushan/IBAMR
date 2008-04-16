define(NDIM,2)dnl
define(REAL,`double precision')dnl
define(INTEGER,`integer')dnl
include(SAMRAI_FORTDIR/pdat_m4arrdim2d.i)dnl
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Interpolate the components of a staggered velocity field onto the
c     faces of the zones centered about the x-component of the velocity.
c
c     NOTE: U0 and U1 are standard side-centered staggered grid
c     velocities for the patch [(ifirst0,ilast0),(ifirst1,ilast1)].  V0
c     and V1 are face-centered staggered grid velocities defined at the
c     faces of the control volumes centered about the x components of
c     the side-centered velocity, i.e., face-centered staggered grid
c     velocities for the patch [(ifirst0,ilast0+1),(ifirst1,ilast1)].
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      subroutine navier_stokes_interp_comps0_2d(
     &     patch_ifirst0,patch_ilast0,patch_ifirst1,patch_ilast1,
     &     n_U_gc0,n_U_gc1,
     &     U0,U1,
     &     side0_ifirst0,side0_ilast0,side0_ifirst1,side0_ilast1,
     &     n_V_gc0,n_V_gc1,
     &     V0,V1)
c
      implicit none
include(TOP_SRCDIR/src/fortran/const.i)dnl
c
c     Input.
c
      INTEGER patch_ifirst0,patch_ilast0
      INTEGER patch_ifirst1,patch_ilast1

      INTEGER n_U_gc0,n_U_gc1

      INTEGER side0_ifirst0,side0_ilast0
      INTEGER side0_ifirst1,side0_ilast1

      INTEGER n_V_gc0,n_V_gc1

      REAL U0(
     &     SIDE2d0VECG(patch_ifirst,patch_ilast,n_U_gc)
     &     )
      REAL U1(
     &     SIDE2d1VECG(patch_ifirst,patch_ilast,n_U_gc)
     &     )
c
c     Input/Output.
c
      REAL V0(
     &     FACE2d0VECG(side0_ifirst,side0_ilast,n_V_gc)
     &     )
      REAL V1(
     &     FACE2d1VECG(side0_ifirst,side0_ilast,n_V_gc)
     &     )
c
c     Local variables.
c
      INTEGER i0,i1
      INTEGER gc0,gc1
c
c     Interpolate the components of the velocity at each zone face.
c
      gc0 = min(n_U_gc0-1,n_V_gc0)
      gc1 = min(n_U_gc1  ,n_V_gc1)

      do    i1 = side0_ifirst1-gc1,side0_ilast1+gc1
         do i0 = side0_ifirst0-gc0,side0_ilast0+gc0+1
            V0(i0,i1) = 0.5d0*(U0(i0-1,i1)+U0(i0,i1))
            V1(i1,i0) = 0.5d0*(U1(i0-1,i1)+U1(i0,i1))
         enddo
      enddo
c
      return
      end
c
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Interpolate the components of a staggered velocity field onto the
c     faces of the zones centered about the y-component of the velocity.
c
c     NOTE: U0 and U1 are standard side-centered staggered grid
c     velocities for the patch [(ifirst0,ilast0),(ifirst1,ilast1)].  V0
c     and V1 are face-centered staggered grid velocities defined at the
c     faces of the control volumes centered about the y components of
c     the side-centered velocity, i.e., face-centered staggered grid
c     velocities for the patch [(ifirst0,ilast0),(ifirst1,ilast1+1)].
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      subroutine navier_stokes_interp_comps1_2d(
     &     patch_ifirst0,patch_ilast0,patch_ifirst1,patch_ilast1,
     &     n_U_gc0,n_U_gc1,
     &     U0,U1,
     &     side1_ifirst0,side1_ilast0,side1_ifirst1,side1_ilast1,
     &     n_V_gc0,n_V_gc1,
     &     V0,V1)
c
      implicit none
include(TOP_SRCDIR/src/fortran/const.i)dnl
c
c     Input.
c
      INTEGER patch_ifirst0,patch_ilast0
      INTEGER patch_ifirst1,patch_ilast1

      INTEGER n_U_gc0,n_U_gc1

      INTEGER side1_ifirst0,side1_ilast0
      INTEGER side1_ifirst1,side1_ilast1

      INTEGER n_V_gc0,n_V_gc1

      REAL U0(
     &     SIDE2d0VECG(patch_ifirst,patch_ilast,n_U_gc)
     &     )
      REAL U1(
     &     SIDE2d1VECG(patch_ifirst,patch_ilast,n_U_gc)
     &     )
c
c     Input/Output.
c
      REAL V0(
     &     FACE2d0VECG(side1_ifirst,side1_ilast,n_V_gc)
     &     )
      REAL V1(
     &     FACE2d1VECG(side1_ifirst,side1_ilast,n_V_gc)
     &     )
c
c     Local variables.
c
      INTEGER i0,i1
      INTEGER gc0,gc1
c
c     Interpolate the components of the velocity at each zone face.
c
      gc0 = min(n_U_gc0  ,n_V_gc0)
      gc1 = min(n_U_gc1-1,n_V_gc1)

      do    i0 = side1_ifirst0-gc0,side1_ilast0+gc0
         do i1 = side1_ifirst1-gc1,side1_ilast1+gc1+1
            V0(i0,i1) = 0.5d0*(U0(i0,i1-1)+U0(i0,i1))
            V1(i1,i0) = 0.5d0*(U1(i0,i1-1)+U1(i0,i1))
         enddo
      enddo
c
      return
      end
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Extract the cell-centered components of the advection velocity.
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      subroutine navier_stokes_extract_cc_velocity2d(
     &     patch_ifirst0,patch_ilast0,patch_ifirst1,patch_ilast1,
     &     n_U_cc_gc0,n_U_cc_gc1,
     &     U_cc,
     &     side0_ifirst0,side0_ilast0,side0_ifirst1,side0_ilast1,
     &     n_U_adv0_gc0,n_U_adv0_gc1,
     &     U_adv00,
     &     side1_ifirst0,side1_ilast0,side1_ifirst1,side1_ilast1,
     &     n_U_adv1_gc0,n_U_adv1_gc1,
     &     U_adv11)
c
      implicit none
include(TOP_SRCDIR/src/fortran/const.i)dnl
c
c     Input.
c
      INTEGER patch_ifirst0,patch_ilast0
      INTEGER patch_ifirst1,patch_ilast1

      INTEGER n_U_cc_gc0,n_U_cc_gc1

      INTEGER side0_ifirst0,side0_ilast0
      INTEGER side0_ifirst1,side0_ilast1

      INTEGER n_U_adv0_gc0,n_U_adv0_gc1

      INTEGER side1_ifirst0,side1_ilast0
      INTEGER side1_ifirst1,side1_ilast1

      INTEGER n_U_adv1_gc0,n_U_adv1_gc1

      REAL U_adv00(
     &     FACE2d0VECG(side0_ifirst,side0_ilast,n_U_adv0_gc)
     &     )
      REAL U_adv11(
     &     FACE2d1VECG(side1_ifirst,side1_ilast,n_U_adv1_gc)
     &     )
c
c     Input/Output.
c
      REAL U_cc(
     &     CELL2dVECG(patch_ifirst,patch_ilast,n_U_cc_gc),
     &     0:NDIM-1)
c
c     Local variables.
c
      INTEGER i0,i1
      INTEGER gc0,gc1
c
c     Extract the cell-centered components of the advection velocity.
c
      gc0 = min(n_U_cc_gc0,n_U_adv0_gc0)
      gc1 = min(n_U_cc_gc1,n_U_adv1_gc1)

      do    i1 = patch_ifirst1-gc1,patch_ilast1+gc1
         do i0 = patch_ifirst0-gc0,patch_ilast0+gc0
            U_cc(i0,i1,0) = U_adv00(i0+1,i1)
            U_cc(i0,i1,1) = U_adv11(i1+1,i0)
         enddo
      enddo
c
      return
      end
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Reset the face-centered advection velocity about the control
c     volumes for the x-component of the velocity.
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      subroutine navier_stokes_reset_adv_velocity0_2d(
     &     side0_ifirst0,side0_ilast0,side0_ifirst1,side0_ilast1,
     &     n_U_adv0_gc0,n_U_adv0_gc1,
     &     U_adv00,U_adv01,
     &     n_U_half0_gc0,n_U_half0_gc1,
     &     U_half00,
     &     side1_ifirst0,side1_ilast0,side1_ifirst1,side1_ilast1,
     &     n_U_half1_gc0,n_U_half1_gc1,
     &     U_half10)
c
      implicit none
include(TOP_SRCDIR/src/fortran/const.i)dnl
c
c     Input.
c
      INTEGER side0_ifirst0,side0_ilast0
      INTEGER side0_ifirst1,side0_ilast1

      INTEGER n_U_adv0_gc0,n_U_adv0_gc1
      INTEGER n_U_half0_gc0,n_U_half0_gc1

      INTEGER side1_ifirst0,side1_ilast0
      INTEGER side1_ifirst1,side1_ilast1

      INTEGER n_U_half1_gc0,n_U_half1_gc1

      REAL U_half00(
     &     FACE2d0VECG(side0_ifirst,side0_ilast,n_U_half0_gc)
     &     )
      REAL U_half10(
     &     FACE2d0VECG(side1_ifirst,side1_ilast,n_U_half1_gc)
     &     )
c
c     Input/Output.
c
      REAL U_adv00(
     &     FACE2d0VECG(side0_ifirst,side0_ilast,n_U_adv0_gc)
     &     )
      REAL U_adv01(
     &     FACE2d1VECG(side0_ifirst,side0_ilast,n_U_adv0_gc)
     &     )
c
c     Local variables.
c
      INTEGER i0,i1
      INTEGER gc0,gc1
c
c     Reset the advection velocity.
c
      gc0 = min(n_U_half0_gc0,n_U_half1_gc0,n_U_adv0_gc0)
      gc1 = min(n_U_half0_gc1,n_U_half1_gc1,n_U_adv0_gc1)

      do    i1 = side0_ifirst1-gc1,side0_ilast1+gc1
         do i0 = side0_ifirst0-gc0,side0_ilast0+gc0
            U_adv00(i0,i1) = U_half00(i0,i1)
            U_adv01(i1,i0) = U_half10(i0,i1)
         enddo
      enddo
c
      return
      end
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Reset the face-centered advection velocity about the control
c     volumes for the y-component of the velocity.
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      subroutine navier_stokes_reset_adv_velocity1_2d(
     &     side0_ifirst0,side0_ilast0,side0_ifirst1,side0_ilast1,
     &     n_U_half0_gc0,n_U_half0_gc1,
     &     U_half01,
     &     side1_ifirst0,side1_ilast0,side1_ifirst1,side1_ilast1,
     &     n_U_adv1_gc0,n_U_adv1_gc1,
     &     U_adv10,U_adv11,
     &     n_U_half1_gc0,n_U_half1_gc1,
     &     U_half11)
c
      implicit none
include(TOP_SRCDIR/src/fortran/const.i)dnl
c
c     Input.
c
      INTEGER side0_ifirst0,side0_ilast0
      INTEGER side0_ifirst1,side0_ilast1

      INTEGER n_U_half0_gc0,n_U_half0_gc1

      INTEGER side1_ifirst0,side1_ilast0
      INTEGER side1_ifirst1,side1_ilast1

      INTEGER n_U_adv1_gc0,n_U_adv1_gc1
      INTEGER n_U_half1_gc0,n_U_half1_gc1

      REAL U_half01(
     &     FACE2d1VECG(side0_ifirst,side0_ilast,n_U_half0_gc)
     &     )
      REAL U_half11(
     &     FACE2d1VECG(side1_ifirst,side1_ilast,n_U_half1_gc)
     &     )
c
c     Input/Output.
c
      REAL U_adv10(
     &     FACE2d0VECG(side1_ifirst,side1_ilast,n_U_adv1_gc)
     &     )
      REAL U_adv11(
     &     FACE2d1VECG(side1_ifirst,side1_ilast,n_U_adv1_gc)
     &     )
c
c     Local variables.
c
      INTEGER i0,i1
      INTEGER gc0,gc1
c
c     Reset the advection velocity.
c
      gc0 = min(n_U_half0_gc0,n_U_half1_gc0,n_U_adv1_gc0)
      gc1 = min(n_U_half0_gc1,n_U_half1_gc1,n_U_adv1_gc1)

      do    i1 = side1_ifirst1-gc1,side1_ilast1+gc1
         do i0 = side1_ifirst0-gc0,side1_ilast0+gc0
            U_adv10(i0,i1) = U_half01(i1,i0)
            U_adv11(i1,i0) = U_half11(i1,i0)
         enddo
      enddo
c
      return
      end
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc