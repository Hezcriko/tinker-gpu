#include "ff/energy.h"
#include "ff/ennintermol.h"
#include "ff/potent.h"
#include "nn/nn.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"

#include <tinker/detail/keys.hh>
#include <tinker/detail/group.hh>
#include <tinker/detail/atomid.hh>
#include <algorithm>
#include <functional>


namespace tinker {

void ennmetalData(RcOp op)
{
   if (not use(Potent::NNMET))
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(grps_nnmetal);

      for (int i=0; i < nnps.size(); i++) {
         if (nnps[i].type == "metal") {
            nnps[i].deallocate();
         }
      }

      if (rc_a)
         bufferDeallocate(rc_flag, ennmet, vir_ennmet, dennmet_x, dennmet_y, dennmet_z);
      ennmet = nullptr;
      vir_ennmet = nullptr;
      dennmet_x = nullptr;
      dennmet_y = nullptr;
      dennmet_z = nullptr;
      darray::deallocate(gx_tmp, gy_tmp, gz_tmp);
   }

   if (op & RcOp::ALLOC) {
      for (int i=0; i < nnterms.size(); i++) {
         if (nnterms[i][0] == "metal") {
            for (int j=1; j < nnterms[i].size(); j++) {
               grps_nnmetal_host.push_back(std::stoi(nnterms[i][j]));
            }
         }
      }
      ngrps_nnmetal = grps_nnmetal_host.size();
      darray::allocate(ngrps_nnmetal, &grps_nnmetal);

      // get the list of atoms in the groups
      std::vector<int> nnatoms;
      for (int i=0; i < n; i++) {
         for (int j=0; j < ngrps_nnmetal; j++){
            if (group::grplist[i] == grps_nnmetal_host[j]) {
               nnatoms.push_back(i);
            }
         }
      }
      nennmet = nnatoms.size();
      // sort nnatoms based on atomic number
      std::sort(nnatoms.begin(), nnatoms.end(), [](int a, int b) {
         return atomid::atomic[a] < atomid::atomic[b];
      });

      // allocate for nnps
      for (int i=0; i < nnps.size(); i++) {
         if (nnps[i].type == "metal") {
            nnps[i].remove_nn_unneeded(nnatoms);
            nnps[i].allocate(nnatoms);
         }
      }

      // test if rc_a, whether eng_buf is always null.
      ennmet = eng_buf;
      vir_ennmet = vir_buf;
      dennmet_x = gx;
      dennmet_y = gy;
      dennmet_z = gz;
      if (rc_a)
         bufferAllocate(rc_flag, &ennmet, &vir_ennmet, &dennmet_x, &dennmet_y, &dennmet_z);
      darray::allocate(n, &gx_tmp, &gy_tmp, &gz_tmp);
   }

   if (op & RcOp::INIT) {
      darray::copyin(g::q0, ngrps_nnmetal, grps_nnmetal, grps_nnmetal_host.data());

      for (int i=0; i < nnps.size(); i++) {
         if (nnps[i].type == "metal") {
            nnps[i].initialize();
         }
      }

      waitFor(g::q0);
   }
}


void ennmetal_cu(int vers)
{
   // auto rc_a = rc_flag & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;
   for (int i = 0; i < nnps.size(); i++){
      if (nnps[i].type == "metal"){
         if (do_e or do_g){
            nnps[i].forward(calc::energy, ngrps_nnmetal, grps_nnmetal, ennmet);
         }
         if (do_g)
            nnps[i].gradient(calc::grad, ngrps_nnmetal, grps_nnmetal, dennmet_x, dennmet_y, dennmet_z, vir_ennmet);
         // TODO implement virial for nnmetal
         // if (do_v)
         //    some code;
      }
   }
}


void ennintermol_cu(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   zeroOnHost(energy_nnintermol);

   bool flag_nnmet = use(Potent::NNMET);
   // in case there will be other intermolecular NN terms

   size_t bsize = bufferSize();
   if (rc_a and flag_nnmet) {
   // if (flag_nnval) {
      zeroOnHost(energy_ennmet, virial_ennmet);
      // if (do_e)
      darray::zero(g::q0, bsize, ennmet);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ennmet);
      if (do_g){
         darray::zero(g::q0, n, dennmet_x, dennmet_y, dennmet_z);
      }
   }

    if (pltfm_config & Platform::CUDA) {
        if (flag_nnmet) {
            ennmetal_cu(vers);
        }
    }

   if (rc_a and flag_nnmet) {
      if (do_e) {
         energy_ennmet = energyReduce(ennmet);
         energy_nnintermol += energy_ennmet;
      }
      if (do_v) {
         // virialReduce(virial_eb, vir_eb);
         // for (int iv = 0; iv < 9; ++iv)
         //    virial_valence[iv] += virial_eb[iv];
      }
      if (do_g)
         sumGradient(gx, gy, gz, dennmet_x, dennmet_y, dennmet_z);
   }
}

}
