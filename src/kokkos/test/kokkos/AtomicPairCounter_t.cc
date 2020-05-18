#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

#include <iostream>
// dirty, but works
#include "KokkosCore/AtomicPairCounter.h"

typedef Kokkos::TeamPolicy<KokkosExecSpace> team_policy;
typedef Kokkos::TeamPolicy<KokkosExecSpace>::member_type  member_type;

void test(){

  Kokkos::View<AtomicPairCounter*,KokkosExecSpace> dc_d("dc_d",1);
  Kokkos::View<AtomicPairCounter*,KokkosExecSpace>::HostMirror dc_h("dc_h",1);

  std::cout << "size " << sizeof(AtomicPairCounter) << std::endl;

  constexpr uint32_t N = 20000;
  constexpr uint32_t M = N * 6;

  Kokkos::View<uint32_t*,KokkosExecSpace> n_d("n_d",N);
  Kokkos::View<uint32_t*,KokkosExecSpace> m_d("m_d",M);
  Kokkos::View<uint32_t*,KokkosExecSpace>::HostMirror n_h("n_h",N);
  Kokkos::View<uint32_t*,KokkosExecSpace>::HostMirror m_h("m_h",M);

  const uint32_t n = 10000;
  Kokkos::parallel_for("update",team_policy(2000,512),
                       KOKKOS_LAMBDA(const member_type &teamMember){
    uint32_t i = teamMember.league_rank() * teamMember.league_size() + teamMember.team_rank();
    if (i >= n)
      return;

    auto m = i % 11;
    m = m % 6 + 1;  // max 6, no 0
    auto c = dc_d(0).add(m);
    assert(c.m < n);
    n_d[c.m] = c.n;
    for (uint32_t j = c.n; j < c.n + m; ++j)
      m_d[j] = i;

   });

  Kokkos::parallel_for("finalize",team_policy(1,1),
                       KOKKOS_LAMBDA(const member_type &teamMember){
    assert(dc_d(0).get().m == n);
    n_d[n] = dc_d(0).get().n;
  });

  Kokkos::parallel_for("verify",team_policy(2000,512),
                       KOKKOS_LAMBDA(const member_type &teamMember){
    uint32_t i = teamMember.league_rank() * teamMember.league_size() + teamMember.team_rank();
    if (i >= n)
      return;
    assert(0 == n_d[0]);
    assert(dc_d(0).get().m == n);
    assert(n_d[n] == dc_d(0).get().n);
    auto ib = n_d[i];
    auto ie = n_d[i + 1];
    auto k = m_d[ib++];
    assert(k < n);
    for (; ib < ie; ++ib)
      assert(m_d[ib] == k);
  });

  Kokkos::deep_copy(dc_h,dc_d);

  std::cout << dc_h(0).get().n << ' ' << dc_h(0).get().m << std::endl;



} // test()
//
//KOKKOS_FUNCTION update(Kokkos::View<AtomicPairCounter*,KokkosExecSpace> dc,
//                       Kokkos::View<uint32_t*,KokkosExecSpace> ind,
//                       Kokkos::View<uint32_t*,KokkosExecSpace> cont,
//                       uint32_t n,
//                       const member_type &teamMember) {
//  auto i = teamMember.league_rank() * teamMember.league_size() + teamMember.team_rank();
//  if (i >= n)
//    return;
//
//  auto m = i % 11;
//  m = m % 6 + 1;  // max 6, no 0
//  auto c = dc->add(m);
//  assert(c.m < n);
//  ind[c.m] = c.n;
//  for (int j = c.n; j < c.n + m; ++j)
//    cont[j] = i;
//};
//
//KOKKOS_FUNCTION void finalize(Kokkos::View<AtomicPairCounter*,KokkosExecSpace> dc,
//                              Kokkos::View<uint32_t*,KokkosExecSpace> ind,
//                              Kokkos::View<uint32_t*,KokkosExecSpace> cont,
//                              uint32_t n,
//                              const member_type &teamMember) {
//  assert(dc->get().m == n);
//  ind[n] = dc->get().n;
//}

//KOKKOS_FUNCTION void verify(Kokkos::View<AtomicPairCounter*,KokkosExecSpace> dc,
//                            Kokkos::View<uint32_t*,KokkosExecSpace> ind,
//                            Kokkos::View<uint32_t*,KokkosExecSpace> cont,
//                            uint32_t n,
//                            const member_type &teamMember) {
//  auto i = teamMember.league_rank() * teamMember.league_size() + teamMember.team_rank();
//  if (i >= n)
//    return;
//  assert(0 == ind[0]);
//  assert(dc->get().m == n);
//  assert(ind[n] == dc->get().n);
//  auto ib = ind[i];
//  auto ie = ind[i + 1];
//  auto k = cont[ib++];
//  assert(k < n);
//  for (; ib < ie; ++ib)
//    assert(cont[ib] == k);
//}

int main(void) {
  kokkos_common::InitializeScopeGuard kokkosGuard;
  test();
  return 0;
}